import itertools
from functools import reduce
from PIL import Image
import numpy as np

import C
import models
from Button import Button

print("\n================== Optical Character Recognition (OCR) ======================")
print("\nLoading OCR model. This might take a few seconds ....")

from models import ModelInfo, MODEL_INFO_CNN
from R import *
from models_handler import ModelsHandler


class Grid:

    def __init__(self, x, y,
                 width, height,
                 rows, cols,
                 marked_color=COLOR_HIGHLIGHT, unmarked_color=BG_DARK,
                 outline_width=GRID_OUTLINE_WIDTH, outline_radius=GRID_OUTLINE_RADIUS,
                 outline_color=COLOR_HIGHLIGHT,
                 mark_state_changed_callback: callable = None,
                 clear_callback: callable = None):

        self.in_rect = pygame.rect.Rect(x, y, width, height)
        self.rows = rows
        self.cols = cols
        self.cell_w = width // self.cols
        self.cell_h = height // self.rows

        self.marked_color = marked_color
        self.unmarked_color = unmarked_color

        self.outline_width = outline_width
        self.outline_radius = outline_radius
        self.outline_color = outline_color
        self.mark_state_changed_callback = mark_state_changed_callback
        self.clear_callback = clear_callback

        self.cells = np.full((rows, cols), 0, dtype=np.uint8)

    # def get_neighbour_indices(self, pos: tuple, upto_step: int = 1, diagonal: bool = True):
    #     return get_neighbour_indices(pos=pos, shape=self.cells.shape, upto_step=upto_step, diagonal=diagonal)

    def is_over(self, x, y) -> bool:
        return self.in_rect.collidepoint(x, y)

    def get_cell_pos(self, x, y) -> tuple | None:
        if not self.is_over(x, y):
            return None
            # raise ValueError(f"Coordinates ({x},{y}) not within the grid bounds {self.rect}")
        return (y - self.in_rect.y) // self.cell_h, (x - self.in_rect.x) // self.cell_w

    def is_marked(self, pos):
        return self.cells[pos]

    def on_mark_state_changed(self, pos, marked):
        if self.mark_state_changed_callback:
            self.mark_state_changed_callback(pos, marked)

    def set_marked(self, pos, marked: bool, set_neighbours: bool = False, diagonal_neighbours: bool = True,
                   neighbour_span: int = 1):
        val = 1 if marked else 0
        self.cells[pos] = val
        self.on_mark_state_changed(pos, marked)

        if set_neighbours:
            nbs = C.get_neighbour_indices(pos, self.cells.shape, upto_step=neighbour_span, diagonal=diagonal_neighbours)
            for n in nbs:
                self.cells[n] = val
                self.on_mark_state_changed(n, marked)

    @property
    def grayscale_image(self) -> Image:
        return Image.fromarray(self.cells * 255, 'L')

    def get_normalized_2d_image(self, shape=models.IMG_SHAPE,
                                resample=Image.Resampling.BILINEAR) -> np.ndarray:
        if not shape or self.cells.shape == shape:
            arr = self.cells.copy()
        else:
            img = self.grayscale_image.resize(size=shape, resample=resample)
            arr = np.array(img) / 255.0
        return arr

    @property
    def normalized_2d_image(self) -> np.ndarray:
        return self.get_normalized_2d_image()

    def draw(self, _win: pygame.Surface):
        if self.outline_width > 0 and self.outline_color:
            pygame.draw.rect(_win, self.outline_color,
                             (self.in_rect.x - self.outline_width, self.in_rect.y - self.outline_width,
                              self.in_rect.width + (self.outline_width * 2),
                              self.in_rect.height + (self.outline_width * 2)),
                             width=self.outline_width, border_radius=self.outline_radius)

        for index in np.ndindex(*self.cells.shape):
            x, y = self.in_rect.x + (index[1] * self.cell_w), self.in_rect.y + (index[0] * self.cell_h)
            pygame.draw.rect(_win, self.marked_color if self.is_marked(index) else self.unmarked_color,
                             (x, y, self.cell_w, self.cell_h))

    @property
    def has_marked_cells(self):
        return 1 in self.cells

    def clear(self):
        self.cells.fill(0)

        if self.clear_callback:
            self.clear_callback()


def draw(_win: pygame.Surface, _grid: Grid, _buttons):
    ww, wh = _win.get_width(), _win.get_height()

    _win.fill(BG_MEDIUM)  # Bg
    _grid.draw(win)  # Grid

    for bt in _buttons:  # Buttons
        bt.draw(_win)

    # status
    if status_text and len(status_text) > 0:
        status_rect = pygame.Rect(WIN_PADX, WIN_PADY, ww - (WIN_PADY * 2), 50)
        pygame.draw.rect(win, TINT_SELF_DARK, status_rect, border_radius=35)

        status = FONT_STATUS.render(status_text, 1, BG_DARK)
        win.blit(status, ((ww - status.get_width()) / 2, status_rect.centery - (status.get_height() / 2)))

    pygame.display.update()


def go_back():
    global run

    if grid.has_marked_cells:
        grid.clear()
    else:
        run = False


def predict():
    global status_text

    if not grid.has_marked_cells:
        print("Nothing to predict! Please draw an alphabet on the canvas ....")
        status_text = "Draw an alphabet first!"
        return

    guess = models.get_class_label(model_handler.predict_single(grid.normalized_2d_image))
    # status_text = f"{model_handler.selected_info.short_label} Prediction: {guess}"
    status_text = f"Prediction: {guess}"
    print(f"{model_handler.selected_info.short_label} Prediction: {guess}")

    if sound_enabled:
        play_predict_sound()


def toggle_sound_enabled():
    global sound_enabled

    sound_enabled = not sound_enabled
    print(f"Sounds {'Enabled' if sound_enabled else 'Disabled'}!")


def handle_keydown(_event):
    if _event.key == pygame.K_ESCAPE:
        go_back()
    elif _event.key in (pygame.K_RETURN, pygame.K_SPACE):
        predict()
    elif _event.key == pygame.K_s:
        toggle_sound_enabled()


def on_mark_state_changed(cell_pos: tuple, marked: bool):
    global status_text

    if marked:
        clear_button.text = CLEAR_BUTTON_TEXT
        status_text = None
        pass
    else:
        has_marks = grid.has_marked_cells

        if has_marks:
            clear_button.text = CLEAR_BUTTON_TEXT
        else:
            on_grid_cleared()


def on_grid_cleared():
    global status_text

    clear_button.text = EXIT_BUTTON_TEXT
    status_text = None


def set_marked(cell_pos: tuple, marked: bool):
    grid.set_marked(cell_pos, marked, set_neighbours=True, diagonal_neighbours=True,
                    neighbour_span=GRID_MARK_NEIGHBOUR_SPAN if marked else GRID_UNMARK_NEIGHBOUR_SPAN)


def handle_grid_on_mouse_down_or_motion(_event=None, motion=False):
    global _last_cell

    buttons = pygame.mouse.get_pressed()
    m_pos = pygame.mouse.get_pos()

    if buttons[0] or buttons[2]:
        cell_pos = grid.get_cell_pos(m_pos[0], m_pos[1])
        mark = buttons[0]
        if cell_pos:
            set_marked(cell_pos, mark)

            # interpolation while motion
            if motion:
                if _last_cell and _last_cell != cell_pos:
                    if _last_cell[0] == cell_pos[0]:
                        for c in range(min(_last_cell[1], cell_pos[1]) + 1, max(_last_cell[1], cell_pos[1])):
                            set_marked((cell_pos[0], c), mark)
                    elif _last_cell[1] == cell_pos[1]:
                        for r in range(min(_last_cell[0], cell_pos[0]) + 1, max(_last_cell[0], cell_pos[0])):
                            set_marked((r, cell_pos[1]), mark)
                    else:
                        pass  # Diagonal interpolation

                _last_cell = cell_pos


def sync_model_buttons_state():
    for _bt in model_buttons:
        _bt.active = _bt.id == model_handler.selected_info.id


def sync_all_buttons_state(_event=None):
    global _last_focused_button

    m_pos = _event.pos if _event else pygame.mouse.get_pos()
    focused_bt = None

    for _bt in get_all_buttons():
        _focus = _bt.is_over(*m_pos)
        _bt.active = _focus or _bt.id == model_handler.selected_info.id

        if _focus:
            focused_bt = _bt

    if focused_bt:
        if not _last_focused_button or _last_focused_button != focused_bt:
            _last_focused_button = focused_bt
            if sound_enabled:
                play_button_sound(hover=True)
    else:
        _last_focused_button = None


def handle_mouse_motion(_event=None):
    handle_grid_on_mouse_down_or_motion(_event, True)

    sync_all_buttons_state(_event)


def handle_mouse_button_down(_event=None):
    global _last_cell

    _last_cell = None  # invalidate
    handle_grid_on_mouse_down_or_motion(_event, False)

    buttons = pygame.mouse.get_pressed()
    m_pos = _event.pos if _event else pygame.mouse.get_pos()

    if buttons[0]:  # left click
        if predict_button.is_over(*m_pos):
            predict()
            return

        if clear_button.is_over(*m_pos):
            go_back()
            return

        _got_model = False
        for bt in model_buttons:
            if bt.is_over(*m_pos):
                model_handler.selected_info = bt.tag
                _got_model = True
                break

        if _got_model:
            sync_model_buttons_state()
            if sound_enabled:
                play_button_sound(hover=False)


def on_model_changed(old_model: ModelInfo | None, new_model: ModelInfo):
    global status_text

    print(f"\nSelected Model: {new_model.long_label} ({new_model.short_label})\n")
    status_text = new_model.long_label


def handle_mouse_button_up(_event=None):
    global _last_cell
    _last_cell = None  # invalidate


def create_model_buttons(win_width, win_height) -> list:
    buttons = []
    button_pad_x = 14
    button_pad_y = 10
    button_corner = 4

    button_hgap = 14

    last_button_x2 = 0
    max_b_height = 0

    for info in model_handler.all_infos:
        _bt = Button(_id=info.id, text=info.display_name, x=last_button_x2 + button_hgap, y=0, pad_x=button_pad_x,
                     pad_y=button_pad_y, corner=button_corner,
                     bg=BG_LIGHT, bg_active=COLOR_HIGHLIGHT,
                     font=FONT_BUTTONS_MEDIUM, text_color=FG_MEDIUM, text_color_active=BG_DARK)

        _bt.tag = info

        buttons.append(_bt)
        last_button_x2 = _bt.x2
        max_b_height = max(max_b_height, _bt.height)

    _dx = ((win_width - (buttons[-1].x2 - buttons[0].x)) // 2) - buttons[0].x  # to center button horizontally

    if max_b_height <= 0:
        max_b_height = reduce(lambda a, b: a if a.height >= b.height else b, buttons).height

    for _bt in buttons:
        _bt.height = max_b_height
        _bt.y = win_height - max_b_height - WIN_PADY
        _bt.x += _dx
    return buttons


def create_clear_and_predict_buttons(win_width, win_height):
    # button_pad_x = 16
    # button_pad_y = 10
    # button_corner = 4
    #
    # _bt = Button(_id=ID_CLEAR_BUTTON, text=EXIT_BUTTON_TEXT, x=0, y=0, pad_x=button_pad_x,
    #              pad_y=button_pad_y, corner=button_corner,
    #              bg=TINT_ENEMY_MEDIUM, bg_active=TINT_ENEMY_DARK,
    #              font=FONT_BUTTONS_MEDIUM, text_color=FG_DARK, text_color_active=BG_DARK)
    #
    # _bt.x = win_width - _bt.width - WIN_PADX
    # _bt.y = win_height - _bt.height - WIN_PADY
    # return _bt
    button_pad_x = 16
    button_pad_y = 10
    button_corner = 4

    pred_bt = Button(_id=ID_PREDICT_BUTTON, text=PREDICT_BUTTON_TEXT, x=0, y=0, pad_x=button_pad_x,
                     pad_y=button_pad_y, corner=button_corner,
                     bg=BG_LIGHT, bg_active=TINT_SELF_DARK,
                     font=FONT_BUTTONS_MEDIUM, text_color=FG_DARK, text_color_active=BG_DARK)

    clear_bt = Button(_id=ID_CLEAR_BUTTON, text=EXIT_BUTTON_TEXT, x=0, y=0, pad_x=button_pad_x,
                      pad_y=button_pad_y, corner=button_corner,
                      bg=BG_LIGHT, bg_active=TINT_ENEMY_DARK,
                      font=FONT_BUTTONS_MEDIUM, text_color=FG_DARK, text_color_active=BG_DARK)

    pred_bt.x = win_width - pred_bt.width - WIN_PADX
    pred_bt.y = win_height - pred_bt.height - WIN_PADY

    clear_bt.x = WIN_PADX
    clear_bt.y = win_height - clear_bt.height - WIN_PADY
    return clear_bt, pred_bt


def get_all_buttons():
    return itertools.chain(model_buttons, (clear_button, predict_button))


_last_cell = None
_last_focused_button = None
status_text = "Draw an alphabet and hit SPACE!"
sound_enabled = DEFAULT_SOUND_ENABLED

model_handler = ModelsHandler(ModelInfo.get_all(),
                              default_model_info=MODEL_INFO_CNN,
                              selected_info_change_callback=on_model_changed,
                              preload=True,
                              preload_in_bg=True)

pygame.init()
pygame.display.init()
pygame.mixer.init()
pygame.font.init()

win_width, win_height = WIN_WIDTH, WIN_HEIGHT
win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Optical Character Recognizer")

clock = pygame.time.Clock()

grid = Grid(GRID_PAD, GRID_PAD, GRID_WIDTH, GRID_HEIGHT, GRID_ROWS, GRID_COLS)
grid.mark_state_changed_callback = on_mark_state_changed  # callback
grid.clear_callback = on_grid_cleared

model_buttons = create_model_buttons(win_width, win_height)
clear_button, predict_button = create_clear_and_predict_buttons(win_width, win_height)

print("\n\tAI Models\n\t\t", "\n\t\t".join((f"{info.short_label} :  {info.long_label}" for info in model_handler.all_infos)))

print("\n\tControls\n\t\tL-Click DRAG :  Draw\n\t\tR-CLick DRAG :  Erase\n\t\tENTER/SPACE :  Recognize drawn "
      "alphabet\n\t\tESCAPE :  Clear canvas / Quit\n\t\tS : Toggle Sound\n")
print("Draw an alphabet on the Canvas to get stared")

run = True

while run:
    clock.tick(FPS)
    draw(win, grid, get_all_buttons())

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break
        elif event.type == pygame.KEYDOWN:
            handle_keydown(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            handle_mouse_button_down(event)
        elif event.type == pygame.MOUSEMOTION:
            handle_mouse_motion(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            handle_mouse_button_up(event)

    if not run:
        break

pygame.quit()
sys.exit(2)
