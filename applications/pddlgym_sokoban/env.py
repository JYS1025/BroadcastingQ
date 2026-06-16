from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from core.env_base import BaseEnv
from core.spaces import DiscreteActionSpace, MultiDiscreteSpace
from core.trainer import AGENT_REGISTRY


AGENT_REGISTRY["sokoban_structural_sbq"] = "applications.pddlgym_sokoban.agent:SokobanStructuralSBQ"


class PDDLGymSokobanEnv(BaseEnv):
    """PDDLGym Sokoban wrapper with fixed-problem object-position factors."""

    ACTION_NAMES = ["up", "down", "left", "right"]

    def __init__(self, config: dict) -> None:
        self.config = config
        env_config = dict(config.get("env", {}))
        observation_config = dict(config.get("observation", {}))
        if str(env_config.get("action_mode", "direction")) != "direction":
            raise ValueError("PDDLGymSokobanEnv supports only env.action_mode='direction'")
        if str(observation_config.get("type", "object_positions")) != "object_positions":
            raise ValueError("PDDLGymSokobanEnv supports only observation.type='object_positions'")

        self.max_steps = env_config.get("max_steps", 100)
        self.max_steps = None if self.max_steps is None else int(self.max_steps)
        self.verbose_info = bool(env_config.get("verbose_info", False))
        self.include_raw_literals_in_info = bool(env_config.get("include_raw_literals_in_info", False))

        try:
            import pddlgym  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PDDLGymSokobanEnv requires pddlgym. Install it manually, for example "
                "`pip install pddlgym`, because repository dependency files are not "
                "modified by this application."
            ) from exc

        self.env = pddlgym.make(str(env_config.get("pddlgym_env_id", "PDDLEnvSokoban-v0")))
        fixed_problem_index = env_config.get("fixed_problem_index", 0)
        if fixed_problem_index is None:
            raise ValueError("PDDLGymSokobanEnv requires env.fixed_problem_index for fixed-size observations")
        self._fix_problem_index(int(fixed_problem_index))

        initial_obs, _ = self._reset_underlying(seed=int(config.get("seed", 0)))
        self._initialize_problem_spec(initial_obs)
        self.action_literals = self._build_direction_actions(initial_obs)
        self.action_names = list(self.ACTION_NAMES)
        self.action_space = DiscreteActionSpace(4)
        self.episode_steps = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        raw_obs, info = self._reset_underlying(seed=seed)
        self.episode_steps = 0
        literals = self._literals(raw_obs)
        obs = self._convert_obs(raw_obs, literals=literals)
        wrapped_info = dict(info)
        wrapped_info.update(self._build_info(obs, literals=literals, action_name=None, success=False))
        return obs, wrapped_info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid Sokoban action {action!r}; expected an integer in [0, 4)")

        result = self.env.step(self.action_literals[int(action)])
        if len(result) == 5:
            raw_obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            raw_obs, reward, terminated, info = result
            truncated = False
        else:
            raise RuntimeError(f"Unexpected PDDLGym step return length: {len(result)}")

        self.episode_steps += 1
        terminated = bool(terminated)
        truncated = bool(truncated)
        if self.max_steps is not None and self.episode_steps >= self.max_steps and not terminated:
            truncated = True

        literals = self._literals(raw_obs)
        obs = self._convert_obs(raw_obs, literals=literals)
        wrapped_info = dict(info)
        wrapped_info.update(
            self._build_info(
                obs,
                literals=literals,
                action_name=self.action_names[int(action)],
                success=bool(terminated),
            )
        )
        return obs, float(reward), terminated, truncated, wrapped_info

    def render(self, mode: str = "human"):
        render = getattr(self.env, "render", None)
        if not callable(render):
            return str(getattr(self, "_last_obs", ""))
        try:
            return render()
        except TypeError:
            return render(mode=mode)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _fix_problem_index(self, index: int) -> None:
        if hasattr(self.env, "fix_problem_index"):
            self.env.fix_problem_index(index)
            return
        if hasattr(self.env, "set_problem_index"):
            self.env.set_problem_index(index)
            return
        if hasattr(self.env, "_problem_idx"):
            self.env._problem_idx = index
            return
        raise RuntimeError(
            "This PDDLGym version does not expose a known fixed-problem-index API; "
            "cannot safely train a fixed-size Sokoban representation."
        )

    def _reset_underlying(self, seed: int | None = None) -> tuple[Any, dict]:
        try:
            result = self.env.reset(seed=seed)
        except TypeError:
            if seed is not None and hasattr(self.env, "seed"):
                self.env.seed(seed)
            result = self.env.reset()

        if isinstance(result, tuple) and len(result) == 2:
            return result[0], dict(result[1])
        return result, {}

    def _initialize_problem_spec(self, obs) -> None:
        literals = self._literals(obs)
        objects = self._objects(obs)
        player = self._find_tagged_object(literals, "is-player", objects, ("player",))
        stones = self._find_tagged_objects(literals, "is-stone", objects, ("stone", "box"))
        locations = self._find_locations(literals, objects, obs)

        if player is None:
            raise RuntimeError("Could not find Sokoban player object from literals or objects")
        if not stones:
            raise RuntimeError("Could not find Sokoban stone objects from literals or objects")
        if not locations:
            raise RuntimeError("Could not find Sokoban location objects from literals or objects")

        self.player_object = player
        self.stone_objects = sorted(stones, key=self._object_name)
        self.location_objects = sorted(locations, key=self._object_name)
        self.player_object_name = self._object_name(self.player_object)
        self.stone_object_names = [self._object_name(obj) for obj in self.stone_objects]
        self.location_names = [self._object_name(obj) for obj in self.location_objects]
        self.goal_location_names = sorted(self._object_name(location) for location in self._goal_locations(obs))
        self.location_to_index = {
            name: idx
            for idx, name in enumerate(self.location_names)
        }
        self.observation_space = MultiDiscreteSpace(
            [len(self.location_objects)] * (1 + len(self.stone_objects))
        )
        self.location_adjacency = self._build_location_adjacency(literals)
        object.__setattr__(self.observation_space, "location_names", list(self.location_names))
        object.__setattr__(self.observation_space, "location_adjacency", tuple(tuple(sorted(v)) for v in self.location_adjacency))

    def _build_location_adjacency(self, literals: Iterable[Any]) -> list[set[int]]:
        adjacency = [set() for _ in self.location_objects]
        for literal in literals:
            if self._normalize(self._predicate_name(literal)) != "move-dir":
                continue
            variables = self._variables(literal)
            if len(variables) < 2:
                continue
            src = self.location_to_index.get(self._object_name(variables[0]))
            dst = self.location_to_index.get(self._object_name(variables[1]))
            if src is None or dst is None:
                continue
            adjacency[src].add(dst)
            adjacency[dst].add(src)
        return adjacency

    def _convert_obs(self, obs, literals: set[Any] | None = None) -> np.ndarray:
        self._last_obs = obs
        literals = self._literals(obs) if literals is None else literals
        at_map = self._at_map(literals)
        values: list[int] = []
        missing: list[str] = []
        object_names = [self.player_object_name] + self.stone_object_names
        for obj_name in object_names:
            loc = at_map.get(obj_name)
            if loc is None:
                missing.append(obj_name)
                continue
            loc_name = self._object_name(loc)
            if loc_name not in self.location_to_index:
                raise RuntimeError(f"Location {loc_name!r} is not in the fixed Sokoban location set")
            values.append(self.location_to_index[loc_name])
        if missing:
            raise RuntimeError(f"Missing at(?thing, ?location) literals for objects: {missing}")

        arr = np.asarray(values, dtype=np.int64)
        if not self.observation_space.contains(arr):
            raise RuntimeError(
                f"Converted Sokoban observation {arr!r} is outside "
                f"MultiDiscreteSpace({self.observation_space.nvec})"
            )
        return arr

    def _build_direction_actions(self, obs) -> list[Any]:
        candidates = self._action_candidates(obs)
        mapping: dict[str, Any] = {}
        for action in candidates:
            pred_name = self._predicate_name(action)
            if "move" not in self._normalize(pred_name):
                continue
            for variable in self._variables(action):
                direction = self._direction_name(variable)
                if direction in self.ACTION_NAMES:
                    mapping.setdefault(direction, action)

        if len(mapping) < 4:
            constructed = self._construct_direction_actions(obs)
            mapping.update({key: value for key, value in constructed.items() if key not in mapping})

        missing = [name for name in self.ACTION_NAMES if name not in mapping]
        if missing:
            raise RuntimeError(
                "Could not build fixed Sokoban direction actions. "
                f"Missing={missing}; available_actions={[str(a) for a in candidates]}; "
                f"objects={[self._object_name(o) for o in self._objects(obs)]}"
            )
        return [mapping[name] for name in self.ACTION_NAMES]

    def _action_candidates(self, obs) -> list[Any]:
        action_space = getattr(self.env, "action_space", None)
        candidates: list[Any] = []
        for attr in ("all_ground_literals", "all_ground_actions"):
            method = getattr(action_space, attr, None)
            if callable(method):
                for args in ((obs,), ()):
                    try:
                        candidates.extend(list(method(*args)))
                        break
                    except TypeError:
                        continue
        for attr in ("literals", "_literals", "all_literals", "_all_ground_literals"):
            value = getattr(action_space, attr, None)
            if value is not None:
                candidates.extend(list(value() if callable(value) else value))

        unique: dict[str, Any] = {}
        for candidate in candidates:
            unique.setdefault(str(candidate), candidate)
        return list(unique.values())

    def _construct_direction_actions(self, obs) -> dict[str, Any]:
        direction_objects = {
            direction: obj
            for obj in self._objects(obs)
            for direction in [self._direction_name(obj)]
            if direction in self.ACTION_NAMES
        }
        move_predicate = self._find_move_predicate()
        if move_predicate is None:
            return {}

        constructed: dict[str, Any] = {}
        for direction, obj in direction_objects.items():
            try:
                constructed[direction] = move_predicate(obj)
            except Exception:
                continue
        return constructed

    def _find_move_predicate(self):
        action_space = getattr(self.env, "action_space", None)
        for attr in ("predicates", "_predicates"):
            predicates = getattr(action_space, attr, None)
            if predicates is None:
                continue
            for predicate in predicates:
                if "move" in self._normalize(getattr(predicate, "name", str(predicate))):
                    return predicate
        return None

    def _build_info(
        self,
        obs: np.ndarray,
        *,
        literals: set[Any] | None,
        action_name: str | None,
        success: bool,
    ) -> dict:
        info = {
            "symbolic_state": {
                "player_location": int(obs[0]),
                **{
                    f"stone_{idx}_location": int(value)
                    for idx, value in enumerate(obs[1:])
                },
            },
            "episode_steps": int(self.episode_steps),
            "success": bool(success),
        }
        if self.verbose_info:
            info.update(
                {
                    "player_object": self.player_object_name,
                    "stone_objects": list(self.stone_object_names),
                    "locations": list(self.location_names),
                    "goal_locations": list(self.goal_location_names),
                }
            )
        if self.include_raw_literals_in_info:
            literals = set() if literals is None else literals
            info["raw_literals"] = sorted(str(literal) for literal in literals)
        if action_name is not None:
            info["action_name"] = action_name
        return info

    def _literals(self, obs) -> set[Any]:
        return set(getattr(obs, "literals", set()))

    def _objects(self, obs) -> set[Any]:
        return set(getattr(obs, "objects", set()))

    def _find_tagged_object(
        self,
        literals: Iterable[Any],
        tag: str,
        objects: Iterable[Any],
        fallback_name_parts: tuple[str, ...],
    ):
        tagged = self._find_tagged_objects(literals, tag, objects, fallback_name_parts)
        return sorted(tagged, key=self._object_name)[0] if tagged else None

    def _find_tagged_objects(
        self,
        literals: Iterable[Any],
        tag: str,
        objects: Iterable[Any],
        fallback_name_parts: tuple[str, ...],
    ) -> set[Any]:
        norm_tag = self._normalize(tag)
        found = {
            self._variables(literal)[0]
            for literal in literals
            if self._normalize(self._predicate_name(literal)) == norm_tag
            and self._variables(literal)
        }
        if found:
            return found
        return {
            obj
            for obj in objects
            if any(part in self._normalize(self._object_name(obj)) for part in fallback_name_parts)
            or any(part in self._normalize(self._object_type(obj)) for part in fallback_name_parts)
        }

    def _find_locations(self, literals: Iterable[Any], objects: Iterable[Any], obs) -> set[Any]:
        locations = {
            obj
            for obj in objects
            if "loc" in self._normalize(self._object_type(obj))
            or "location" in self._normalize(self._object_type(obj))
        }
        for literal in literals:
            pred = self._normalize(self._predicate_name(literal))
            variables = self._variables(literal)
            if pred == "at" and len(variables) >= 2:
                locations.add(variables[1])
            if "goal" in pred:
                locations.update(self._goal_location_candidates(variables))
        locations.update(self._goal_locations(obs))
        return locations

    def _at_map(self, literals: Iterable[Any]) -> dict[str, Any]:
        at_map: dict[str, Any] = {}
        for literal in literals:
            if self._normalize(self._predicate_name(literal)) != "at":
                continue
            variables = self._variables(literal)
            if len(variables) >= 2:
                at_map[self._object_name(variables[0])] = variables[1]
        return at_map

    def _goal_locations(self, obs) -> set[Any]:
        locations: set[Any] = set()
        for literal in self._flatten_goal_literals(getattr(obs, "goal", None)):
            pred = self._normalize(self._predicate_name(literal))
            variables = self._variables(literal)
            if pred == "at" and len(variables) >= 2:
                locations.add(variables[1])
            elif "goal" in pred:
                locations.update(self._goal_location_candidates(variables))
        for literal in self._literals(obs):
            if "goal" in self._normalize(self._predicate_name(literal)):
                locations.update(self._goal_location_candidates(self._variables(literal)))
        return locations

    def _goal_location_candidates(self, variables: tuple[Any, ...]) -> set[Any]:
        if not variables:
            return set()
        if len(variables) == 1:
            return {variables[0]}
        return {variables[-1]}

    def _flatten_goal_literals(self, goal) -> list[Any]:
        if goal is None:
            return []
        if hasattr(goal, "literals"):
            return list(goal.literals)
        if isinstance(goal, (set, list, tuple, frozenset)):
            out: list[Any] = []
            for item in goal:
                out.extend(self._flatten_goal_literals(item))
            return out
        return [goal]

    def _predicate_name(self, literal) -> str:
        predicate = getattr(literal, "predicate", None)
        return str(getattr(predicate, "name", predicate if predicate is not None else ""))

    def _variables(self, literal) -> tuple[Any, ...]:
        return tuple(getattr(literal, "variables", getattr(literal, "terms", ())))

    def _object_name(self, obj) -> str:
        return str(getattr(obj, "name", obj))

    def _object_type(self, obj) -> str:
        for attr in ("var_type", "type_name", "type", "typename"):
            value = getattr(obj, attr, None)
            if value is not None:
                return str(value)
        return ""

    def _direction_name(self, obj) -> str | None:
        name = self._normalize(self._object_name(obj))
        for direction in self.ACTION_NAMES:
            if name == direction or name.endswith(direction) or direction in name.split("-"):
                return direction
        return None

    @staticmethod
    def _normalize(value: str) -> str:
        return str(value).lower().replace("_", "-").replace(":", "-")
