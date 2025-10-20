from typing import List
from .entities import Entity

class World:
    """
    Manages the state of all entities in the simulation.

    It acts as a container for all objects and provides a single entry point
    to advance the state of the entire simulation.
    """
    def __init__(self):
        self.entities: List[Entity] = []

    def add_entity(self, entity: Entity):
        """
        Adds an entity to the world.
        This is how you populate your scene.
        """
        self.entities.append(entity)

    def update(self, dt: float):
        """
        Updates every entity in the world for a given time step 'dt'.
        The World delegates the update logic to each individual entity.
        """
        for entity in self.entities:
            entity.update(dt)
    
    def clear_entities(self):
        """Removes all entities from the world."""
        self.entities.clear()
