import os
import sys

import pygame
import sklearn
from pygame import Color

# ...........................  File Paths  ......................................
FROZEN = getattr(sys, 'frozen', False)
DIR_MAIN = os.path.dirname(sys.executable) if FROZEN else (
    os.path.dirname(os.path.abspath(os.path.realpath(__file__))))

DIR_MODELS = os.path.join(DIR_MAIN, "models")
DIR_RES = os.path.join(DIR_MAIN, "res")
DIR_RES_IMAGES = os.path.join(DIR_RES, "images")
DIR_RES_SOUND = os.path.join(DIR_RES, "sound")
DIR_RES_FONT = os.path.join(DIR_RES, "font")


def get_model_file_path(file_name: str):
    return os.path.join(DIR_MODELS, file_name)


# ..........................  Models  ..............................

def save_sklearn_model(model, file_name: str, compression_level: int = 2):
    """
    :param model: model to save
    :param file_name: name of the model
    :param compression_level: level of compression, in range of 0-9
    """
    print(f"Saving sklearn model {file_name} ...")
    sklearn.utils._joblib.dump(model, get_model_file_path(file_name), compress=compression_level)


def load_sklearn_model(file_name: str):
    """
    :return loaded model
    """
    return sklearn.utils._joblib.load(get_model_file_path(file_name))


def save_keras_model(model, file_name):
    print(f"Saving keras model {file_name} ...")
    model.save(get_model_file_path(file_name))


def load_keras_model(file_name):
    from tensorflow import keras

    return keras.models.load_model(get_model_file_path(file_name))


# .............................  UI ...........................

def gray(i) -> Color:
    return Color(i, i, i)


WIN_WIDTH, WIN_HEIGHT = 560, 620
GRID_PAD = 0
GRID_WIDTH = min(WIN_WIDTH, WIN_HEIGHT) - (GRID_PAD * 2)
GRID_HEIGHT = GRID_WIDTH
GRID_OUTLINE_WIDTH = 2
GRID_OUTLINE_RADIUS = 2

WIN_PADX = 10
WIN_PADY = 10

GRID_MARK_NEIGHBOUR_SPAN = 1
GRID_UNMARK_NEIGHBOUR_SPAN = 2

FPS = 120

GRID_ROWS = 56
GRID_COLS = 56

# Colors
WHITE = gray(255)
BLACK = gray(0)

BG_DARK = BLACK
BG_MEDIUM = gray(20)
BG_LIGHT = gray(40)

FG_DARK = WHITE
FG_MEDIUM = gray(225)
FG_LIGHT = gray(190)

# COLOR_ACCENT_DARK = Color(57, 206, 255)
# COLOR_ACCENT_MEDIUM = Color(33, 187, 255)
# COLOR_ACCENT_LIGHT = Color(2, 169, 255)

COLOR_ACCENT_DARK = Color(29, 255, 0)
COLOR_ACCENT_MEDIUM = Color(29, 226, 0)
COLOR_ACCENT_LIGHT = Color(29, 190, 0)

COLOR_HIGHLIGHT = Color(253, 255, 52)

COLOR_TRANSPARENT = Color(0, 0, 0, 0)
COLOR_TRANSLUCENT = Color(0, 0, 0, 125)

TINT_SELF_DARK = Color(120, 255, 120)
TINT_SELF_MEDIUM = Color(55, 255, 55)
TINT_SELF_LIGHT = Color(0, 255, 0)

TINT_ENEMY_DARK = Color(255, 120, 120)
TINT_ENEMY_MEDIUM = Color(255, 55, 55)
TINT_ENEMY_LIGHT = Color(255, 0, 0)

ID_CLEAR_BUTTON = 0xFF0AC
CLEAR_BUTTON_TEXT = "Clear"
EXIT_BUTTON_TEXT = "Quit"

ID_PREDICT_BUTTON = 0xFAFB
PREDICT_BUTTON_TEXT = "Predict"

# Fonts
FILE_PATH_FONT_PD_SANS = os.path.join(DIR_RES_FONT, 'product_sans_regular.ttf')
FILE_PATH_FONT_PD_SANS_MEDIUM = os.path.join(DIR_RES_FONT, 'product_sans_medium.ttf')
FILE_PATH_FONT_PD_SANS_LIGHT = os.path.join(DIR_RES_FONT, 'product_sans_light.ttf')
FILE_PATH_FONT_AQUIRE = os.path.join(DIR_RES_FONT, 'aquire.otf')
FILE_PATH_FONT_AQUIRE_LIGHT = os.path.join(DIR_RES_FONT, 'aquire_light.otf')
FILE_PATH_FONT_AQUIRE_BOLD = os.path.join(DIR_RES_FONT, 'aquire_bold.otf')

pygame.font.init()
# FONT_TITLE = pygame.font.Font(FILE_PATH_FONT_AQUIRE, 30)
# FONT_SUMMARY = pygame.font.Font(FILE_PATH_FONT_PD_SANS_LIGHT, 19)
FONT_STATUS = pygame.font.Font(FILE_PATH_FONT_PD_SANS_LIGHT, 20)
FONT_BUTTONS = pygame.font.Font(FILE_PATH_FONT_AQUIRE, 18)
FONT_BUTTONS_MEDIUM = pygame.font.Font(FILE_PATH_FONT_AQUIRE, 14)
FONT_BUTTONS_SMALL = pygame.font.Font(FILE_PATH_FONT_PD_SANS, 13)

# Sound

DEFAULT_SOUND_ENABLED = True

FILE_PATH_SOUND_BUTTON_HOVER = os.path.join(DIR_RES_SOUND, "button_hover.wav")
FILE_PATH_SOUND_BUTTON_CLICK = os.path.join(DIR_RES_SOUND, "button_click.wav")
FILE_PATH_SOUND_PREDICT = os.path.join(DIR_RES_SOUND, "predict.wav")

pygame.mixer.init()
SOUND_BUTTON_HOVER = pygame.mixer.Sound(FILE_PATH_SOUND_BUTTON_HOVER)
SOUND_BUTTON_CLICK = pygame.mixer.Sound(FILE_PATH_SOUND_BUTTON_CLICK)
SOUND_PREDICT = pygame.mixer.Sound(FILE_PATH_SOUND_PREDICT)


def play_button_sound(hover: bool):
    (SOUND_BUTTON_HOVER if hover else SOUND_BUTTON_CLICK).play()


def play_predict_sound():
    SOUND_PREDICT.play()
