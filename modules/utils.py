"""
General purpose utility functions.

------------------------------------------------------------------
Part of programm for photoacoustic measurements using experimental
setup in BioNanoPhotonics lab., NRNU MEPhI, Moscow, Russia.

Author: Anton Popov
contact: a.popov.fizte@gmail.com
            
Created with financial support from Russian Scince Foundation.
Grant # 22-72-00015

2024
"""
import logging

from PySide6.QtCore import (
    QObject
)
from PySide6.QtWidgets import (
    QPushButton,
    QLayout,
    QWidget
)

from pint.facets.plain.quantity import PlainQuantity
import numpy as np

from .gui.widgets import (
    QuantSpinBox
)

logger = logging.getLogger(__name__)

def form_quant(quant: PlainQuantity) -> str:
    """Format str representation of a quantity."""

    return f'{quant:~.2gP}'

def btn_set_silent (btn: QPushButton, state: bool) -> None:
    """Set state of a button without triggering signals."""

    btn.blockSignals(True)
    btn.setChecked(state)
    btn.blockSignals(False)

def __own_properties(cls: type) -> list[str]:
    return [
        key
        for key, value in cls.__dict__.items()
        if isinstance(value, property)
    ]

def properties(cls: type) -> list[str]:
    """Return list of all properties of a class."""
    props = []
    for kls in cls.mro():
        props += __own_properties(kls)
    
    return props

def propvals(instance: object) -> dict:
    """Return values off all properties of a class."""

    props = properties(type(instance))

    return {prop: getattr(instance, prop) for prop in props}

def set_layout_enabled(
        layout: QLayout,
        enabled: bool
    ) -> None:
    """Set enabled for all QWidgets in a layout."""

    for i in range(layout.count()):
        child = layout.itemAt(i)
        if child.layout() is not None:
            set_layout_enabled(child.layout(), enabled)
        else:
            child.widget().setEnabled(enabled) # type: ignore

def set_sb_silent(
        sb: QuantSpinBox,
        value: PlainQuantity
    ) -> None:
    """Set ``value`` to ``sb`` without firing events."""

    sb.blockSignals(True)
    sb.quantity = value.to(sb.quantity.u)
    sb.blockSignals(False)

def get_layout(widget: QWidget) -> QLayout|None:
        """Return parent layout for a widget."""

        parent = widget.parent()
        child_layouts = _unpack_layout(parent.children())
        for layout in child_layouts:
            if layout.indexOf(widget) > -1:
                return layout

def _unpack_layout(childs: list[QObject]) -> list[QLayout]:
        """Recurrent function to find all layouts in a list of widgets."""

        result = []
        for item in childs:
            if isinstance(item, QLayout):
                result.append(item)
                result.extend(_unpack_layout(item.children()))
        return result