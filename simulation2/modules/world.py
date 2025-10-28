from typing import List, Optional

from .entity import Entity
from .camera import Camera

class World:
    """ Manages entities, time, and holds the main camera reference. """
    def __init__(self, delta_t: float = (1.0 / 30.0)):
        self.entities: List[Entity] = []
        self.current_time: float = 0.0
        self.delta_t: float = delta_t
        self.camera: Optional[Camera] = None # Hold the main camera

    def add_entity(self, entity: Entity):
        """ Adds entity and sets camera reference if applicable. """
        self.entities.append(entity)
        if isinstance(entity, Camera) and self.camera is None:
            print(f"Setting main camera to: {entity}")
            self.camera = entity
        elif isinstance(entity, Camera) and self.camera is not None:
             print("Warning: Adding a second camera, but only the first one will be used by default.")

    def step(self):
        """ Advances the simulation by one time step. """
        for entity in self.entities:
            entity.update(self.delta_t)
        self.current_time += self.delta_t