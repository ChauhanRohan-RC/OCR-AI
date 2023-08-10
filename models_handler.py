from collections.abc import Iterable
import numpy as np
import C
from models import ModelInfo


class ModelsHandler:
    DEFAULT_PRELOAD = True
    DEFAULT_PRELOAD_IN_BG = True

    def __init__(self, model_infos: Iterable[ModelInfo],
                 default_model_info: ModelInfo,
                 selected_info_change_callback: callable = None,        # called with old and newly selected model info
                 preload: bool = DEFAULT_PRELOAD,
                 preload_in_bg: bool = DEFAULT_PRELOAD_IN_BG):

        self._model_infos = model_infos
        self._id_to_info = {}
        for info in self._model_infos:
            self._id_to_info[info.id] = info

        self._info_to_model = {}  # model_info -> model

        self.default_model_info: ModelInfo = default_model_info
        self._selected_model_info = default_model_info
        self.selected_info_change_callback = selected_info_change_callback

        if preload:
            self.preload(in_bg=preload_in_bg)

    @property
    def all_infos(self):
        return list(self._model_infos)

    def model_info_from_id(self, _id: int):
        return self._id_to_info[_id]

    @property
    def selected_info(self) -> ModelInfo:
        return self._selected_model_info

    @selected_info.setter
    def selected_info(self, model_info: ModelInfo):
        prev = self._selected_model_info
        if prev == model_info:
            return

        self._selected_model_info = model_info
        if self.selected_info_change_callback:
            self.selected_info_change_callback(prev, model_info)

    def _do_load_model(self, model_info: ModelInfo):
        model = model_info.load_model()
        self._info_to_model[model_info] = model
        return model

    def get_model_from_info(self, model_info: ModelInfo):
        if model_info in self._info_to_model:
            return self._info_to_model[model_info]

        return self._do_load_model(model_info)

    def get_model_from_id(self, _id: int):
        return self.get_model_from_info(self.model_info_from_id(_id))

    def preload(self, in_bg: bool = DEFAULT_PRELOAD_IN_BG):
        for info in self._model_infos:
            if info not in self._info_to_model:
                if in_bg:
                    C.execute_on_thread(self._do_load_model, args=(info,))
                else:
                    self._do_load_model(info)

    def predict(self, images_array: np.ndarray) -> np.ndarray:
        info = self.selected_info
        model = self.get_model_from_info(info)
        return info.predict(model, images_array)

    def predict_single(self, image: np.ndarray):
        return self.predict(image.reshape(1, *image.shape))[0]
