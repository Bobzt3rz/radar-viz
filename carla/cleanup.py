import carla
import time

def main():
    client = None
    try:
        # 1. Connect to the server
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()

        # 2. Define the blueprint IDs of the sensors to clean up
        sensor_types_to_destroy = [
            'sensor.camera.rgb',
            'sensor.other.radar'
        ]

        # 3. Get all actors in the world and filter for our sensors
        actor_list = world.get_actors()
        actors_to_destroy = []

        for actor in actor_list:
            if actor.type_id in sensor_types_to_destroy:
                actors_to_destroy.append(actor)

        if not actors_to_destroy:
            print('No orphaned sensors found. All clean!')
            return

        print(f'Found {len(actors_to_destroy)} sensors to destroy:')

        # 4. Stop any active listeners (to prevent warnings)
        for actor in actors_to_destroy:
            # Check if it's a sensor and is listening
            if hasattr(actor, 'is_listening') and actor.is_listening:
                print(f'  - Stopping listener on: {actor.type_id} (ID: {actor.id})')
                actor.stop()
            else:
                print(f'  - Found: {actor.type_id} (ID: {actor.id})')

        # 5. Destroy them all in a batch
        client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])
        
        print(f'\nSuccessfully destroyed {len(actors_to_destroy)} actors.')

    except Exception as e:
        print(f'An error occurred: {e}')
    
    finally:
        print('Cleanup script finished.')

if __name__ == '__main__':
    main()