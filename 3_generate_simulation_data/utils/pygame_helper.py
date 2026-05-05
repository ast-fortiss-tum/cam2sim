import queue

import pygame
from PIL import Image


def pil_to_surface(pil_img):
    """Konvertiert PIL Image in Pygame Surface"""
    mode = pil_img.mode
    size = pil_img.size
    data = pil_img.tobytes()

    return pygame.image.fromstring(data, size, mode)

def combine_images(image1, image2):
    size_x, size_y = image1.size
    combined = Image.new("RGB", (size_x * 2, size_y))
    combined.paste(image1, (0, 0))
    combined.paste(image2, (size_x, 0))
    return combined

def show_image(pygame_screen, image):
    surface = pil_to_surface(image)
    pygame_screen.blit(surface, (0, 0))
    pygame.display.flip()

def setup_pygame(size_x, size_y, only_carla=False):
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((size_x * 2 if not only_carla else size_x, size_y))
    pygame.display.set_caption("Original | Generated" if not only_carla else "Original")
    return screen, clock

def setup_sensor(sensor_bp, transform, vehicle, world):
    sensor = world.spawn_actor(sensor_bp, transform, attach_to=vehicle)
    q = queue.Queue()
    sensor.listen(q.put)
    return sensor, q

def get_sensor_blueprint(blueprint_library, name, sensor_size_x,sensor_size_y, sensor_size_fov):
    sensor_bp = blueprint_library.find(name)
    sensor_bp.set_attribute('image_size_x', str(sensor_size_x))
    sensor_bp.set_attribute('image_size_y', str(sensor_size_y))
    sensor_bp.set_attribute('fov', str(sensor_size_fov))
    sensor_bp.set_attribute('sensor_tick', '0.0')
    return sensor_bp