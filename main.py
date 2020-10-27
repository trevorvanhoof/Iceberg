# TODO: When pushing an undo command of the same class, see if they can be merged (flatten sequential value edits)
# TODO: When undoing a selection change, undo until a non-selection change was undone (can't be done with undo groups, see next item)
# TODO: When redoing a non-selection change, redo until the next non-selection change is about to be redone (can't be done with undo groups, see previous item)
"""
Key binding changes from Rocket:
I: Enumerate interpolation mode -> We will have many more interpolation modes, so I intend to use CTRL+[0-9] for this instead
Space: Pause/Resume demo -> We don't do playback, animation data is sampled implicitly, playback can be done in the host framework
Ctrl+Left/Right: Move track -> Changed to Alt+Up/Down because our tracker is transposed
E: Insert key with the evaluated value, for quick insert & bias at specific time points
"""

import bisect
import functools
from math import floor
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from typing import *
import enum


class TangentMode(enum.IntEnum):
    Step = 0
    Linear = 1
    Flat = 2
    Spline = 3
    Plateau = 4
    Default = 4


Key = Tuple[float, TangentMode, TangentMode]


class IcebergDict(object):
    def __init__(self):
        self.__internal: Dict[int, Key] = {}
        self.__sortedKeys: Tuple[int] = tuple()
        self.__dirty: bool = True

    def sortedKeys(self):
        if self.__dirty:
            self.__sortedKeys = tuple(sorted(self.__internal.keys()))
            self.__dirty = False
        return self.__sortedKeys

    def values(self):
        return self.__internal.values()

    def __len__(self):
        return len(self.__internal)

    def get(self, column: int, fallback: Key):
        assert isinstance(fallback, tuple)
        assert isinstance(fallback[0], float)
        assert isinstance(fallback[1], TangentMode)
        assert isinstance(fallback[2], TangentMode)
        return self.__internal.get(column, fallback)

    def __contains__(self, column):
        return column in self.__internal

    def __getitem__(self, column: int):
        assert isinstance(column, int)
        return self.__internal[column]

    def __setitem__(self, column: int, value: Key):
        assert isinstance(column, int)
        assert isinstance(value, tuple)
        assert isinstance(value[0], float)
        assert isinstance(value[1], TangentMode)
        assert isinstance(value[2], TangentMode)
        self.__internal[column] = value
        self.__dirty = True

    def __delitem__(self, column: int):
        if column in self.__internal:
            del self.__internal[column]
            self.__dirty = True


class Track(object):
    def __init__(self, name: str, color: QColor):
        self.name: str = name
        self.color: QColor = color
        self.keys = IcebergDict()  # : Dict[int, Key] = {}


class Song(object):
    def __init__(self):
        self.measures: int = 40
        self.measureDivisions: int = 128
        # key is track index, value is dict; value key is time
        self.tracks: List[Track] = []
        self.bookmarks: Dict[int, str] = {}

    def noteCount(self):
        return self.measures * self.measureDivisions

    def evaluateTrack(self, track: IcebergDict, column: float) -> float:
        # no data, early out with 0
        if not track:
            return 0.0
        # short track, early out with the only possible value
        if len(track) == 1:
            return next(iter(track.values()))[0]
            # figure out which curve segment to evaluate
        keys: List[int] = track.sortedKeys()  # TODO: cache this by making track a smarter type that keeps both a list of sorted keys and a dict internally
        nxt: int = bisect.bisect_right(keys, column)
        # if column is out of bounds just return the right endpoint
        if nxt == 0:
            return track[keys[0]][0]
        if nxt >= len(track):
            return track[keys[-1]][0]
        # get the curve segment to evaluate
        prev: int = nxt - 1
        nextCol: int = keys[nxt]
        prevCol: int = keys[prev]
        prevVal, _, prevOTM = track[prevCol]
        nextVal, nextITM, _ = track[nextCol]

        # special case evaluating the start or end of a pattern
        # find when the pattern ends
        bookmarks = sorted(list(self.bookmarks.keys()))  # TODO: cache this by making bookmarks a smarter type that keeps both a list of sorted keys and a dict internally
        nxtB = bisect.bisect_right(bookmarks, column)
        # default to entire song
        patternStart, patternEnd = 0, self.noteCount()
        # there is a pattern after this
        if nxtB < len(bookmarks):
            patternEnd = bookmarks[nxtB]
            # the curve segment overlaps a pattern boundary, return the (pattern-local) endpoint
            if nextCol >= patternEnd:
                return prevVal
        if nxtB > 0:
            patternStart = bookmarks[nxtB - 1]
            # the curve segment overlaps a pattern boundary, return the (pattern-local) start
            if prevCol < patternStart:
                return nextVal

        # special case step curves
        if prevOTM == TangentMode.Step:
            return prevVal

        # define hermite interpolation factors
        # TODO: We should cache oty and ity results and recompute only when modifying (key-local & adjacent) key values or (key-local) tangent modes.
        #  We will need to handle bookmark changes as well, may be lazy about this and recalc all though.
        #  A demo runtime should probably precalc these based on the mode as they are only 3 bits each (as opposed to to storing two 12 or 16 bit floats).
        # note that I will mention vectors but hermite interpolation actually just uses slopes in the end
        # flat tangents are just a flat vector, y = 0
        oty = 0.0
        # plateau selects flat or spline based on whether it is continuous or changing direction; (0, [1], 0.5) = flat, (0, [0.25], 1) = spline
        if prev != 0 and prevOTM == TangentMode.Plateau:
            prev2Col = keys[prev - 1]
            a, b, c = track[prev2Col][0], prevVal, nextVal
            if a < b < c or a > b > c:
                prevOTM = TangentMode.Spline
            else:
                prevOTM = TangentMode.Flat
        # spline tangents are the vector between the previous and the next key, rescaled based on the duration of the curve segment we're evaluating
        if prevOTM == TangentMode.Spline:
            # spline falls back to linear at the curve boundaries
            if prev == 0:  # song boundary
                prevOTM = TangentMode.Linear
            else:
                prev2Col = keys[prev - 1]
                if prev2Col < patternStart:  # pattern boundary
                    prevOTM = TangentMode.Linear
                else:
                    oty = ((nextVal - track[prev2Col][0]) / (nextCol - prev2Col)) * (nextCol - prevCol)  # (prevCol - prev2Col)
        # linear tangents are the vector to the other endpoint of the curve segments
        if prevOTM == TangentMode.Linear:
            oty = nextVal - prevVal

        # flat tangents are just a flat vector, y = 0
        ity = 0.0
        # plateau selects flat or spline based on whether it is continuous or changing direction
        if nxt != len(track) - 1 and nextITM == TangentMode.Plateau:
            next2Col = keys[nxt + 1]
            a, b, c = prevVal, nextVal, track[next2Col][0]
            if a < b < c or a > b > c:
                nextITM = TangentMode.Spline
            else:
                nextITM = TangentMode.Flat
        # spline tangents are the vector between the previous and the next key, rescaled based on the duration of the curve segment we're evaluating
        if nextITM == TangentMode.Spline:
            # spline falls back to linear at the curve boundaries
            if nxt == len(track) - 1:  # song boundary
                nextITM = TangentMode.Linear
            else:
                next2Col = keys[nxt + 1]
                if next2Col >= patternEnd:  # pattern boundary
                    nextITM = TangentMode.Linear
                else:
                    ity = ((track[next2Col][0] - prevVal) / (next2Col - prevCol)) * (nextCol - prevCol)  # (next2Col - nextCol)
        # linear tangents are the vector to the other endpoint of the curve segments
        if nextITM == TangentMode.Linear:
            ity = nextVal - prevVal

        # cubic hermite interpolation
        dx = nextCol - prevCol
        dy = nextVal - prevVal
        c0 = (oty + ity - dy - dy)
        c1 = (dy + dy + dy - oty - oty - ity)
        c2 = oty
        c3 = prevVal
        t = (column - prevCol) / dx
        return t * (t * (t * c0 + c1) + c2) + c3


class AddTrack(QUndoCommand):
    def __init__(self, song, at, track):
        super().__init__(f'Add track "{track.name}"')
        self._song = song
        self._track = track
        self._at = at

    def redo(self):
        self._song.tracks.insert(self._at, self._track)

    def undo(self):
        assert self._song.tracks.pop(self._at) == self._track


class DeleteTracks(QUndoCommand):
    def __init__(self, song, *at):
        super().__init__(f'Delete {len(at)} tracks')
        self._song = song
        self._at = tuple(reversed(sorted(at)))
        self._track = tuple(song.tracks[at] for at in self._at)
        self._data = tuple(song.tracks[at] for at in self._at)

    def redo(self):
        for i, at in enumerate(self._at):
            assert self._song.tracks.pop(at) == self._track[i]

    def undo(self):
        for i, at in enumerate(self._at):
            self._song.tracks.insert(at, self._track[i])


class SetSelection(QUndoCommand):
    # Note: this assumes ownership of the given state
    def __init__(self, ui, state):
        super().__init__('Set selection')
        self._ui = ui
        self._new = state
        self._old = ui.selectionCorners

    def redo(self):
        self._ui.selectionCorners = self._new

    def undo(self):
        self._ui.selectionCorners = self._old


class ResetSelection(QUndoCommand):
    def __init__(self, ui):
        super().__init__('Reset selection')
        self._ui = ui
        self._old = self.selectionCorners

    def redo(self):
        self._ui.selectionCorners = [[0, 0], [0, 0]]

    def undo(self):
        self._ui.selectionCorners = self._old


class SetTangentMode(QUndoCommand):
    def __init__(self, config):
        super().__init__(f'Set tangent mode')
        self._config = config
        self._restore = {}
        for track, delta in self._config.items():
            for (j, _, _) in delta:
                self._restore.setdefault(track, []).append((j, track.keys[j][1], track.keys[j][2]))

    def redo(self):
        for track, delta in self._config.items():
            for (j, it, ot) in delta:
                track.keys[j] = track.keys[j][0], it, ot

    def undo(self):
        for track, delta in self._restore.items():
            for (j, it, ot) in delta:
                track.keys[j] = track.keys[j][0], it, ot


class SetValue(QUndoCommand):
    def __init__(self, config):
        super().__init__(f'Set values')
        self._config = config
        self._restore = {}
        for track, delta in self._config.items():
            for j, _ in delta:
                row = self._restore.setdefault(track, [])
                if j in track.keys:
                    row.append((j, track.keys[j][0]))
                else:
                    row.append((j, None))

    def redo(self):
        for track, delta in self._config.items():
            for j, v in delta:
                if j in track.keys:
                    if v is not None:
                        track.keys[j] = v, track.keys[j][1], track.keys[j][2]
                    else:
                        del track.keys[j]
                elif v is not None:
                    track.keys[j] = v, TangentMode.Default, TangentMode.Default

    def undo(self):
        for track, delta in self._config.items():
            for j, v in delta:
                if v is not None:
                    track.keys[j] = v, track.keys[j][1], track.keys[j][2]
                else:
                    del track.keys[j]


class SetKey(QUndoCommand):
    def __init__(self, config):
        super().__init__(f'Set keys')
        self._config = config
        self._restore = {}
        for track, delta in self._config.items():
            for j, _ in delta:
                row = self._restore.setdefault(track, [])
                if j in track.keys:
                    row.append((j, track.keys[j]))
                else:
                    row.append((j, None))

    def redo(self):
        for track, delta in self._config.items():
            for j, v in delta:
                if j in track.keys:
                    if v is not None:
                        track.keys[j] = v
                    else:
                        del track.keys[j]
                elif v is not None:
                    track.keys[j] = v

    def undo(self):
        for track, delta in self._config.items():
            for j, v in delta:
                if v is not None:
                    track.keys[j] = v
                else:
                    del track.keys[j]


class SetBookmark(QUndoCommand):
    def __init__(self, bookmarks, at, name):
        super(SetBookmark, self).__init__('Set/unset pattern start')
        self._bookmarks = bookmarks
        self._at = at
        self._name = name
        self._oldName = self._bookmarks.get(self._at, '')

    def redo(self):
        if self._name:
            self._bookmarks[self._at] = self._name
        else:
            del self._bookmarks[self._at]

    def undo(self):
        if self._oldName:
            self._bookmarks[self._at] = self._oldName
        else:
            del self._bookmarks[self._at]


class IcebergSpinBox(QDoubleSpinBox):
    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.hide()
            return
        super(IcebergSpinBox, self).keyReleaseEvent(event)


class IcebergTracker(QWidget):
    def __init__(self, undoStack: Optional[QUndoStack] = None):
        super(IcebergTracker, self).__init__()
        self.setObjectName('Iceberg')
        self.setFocusPolicy(Qt.StrongFocus)

        self.undoStack: QUndoStack = undoStack or QUndoStack()
        self.undoStack.indexChanged.connect(self.repaint)

        self.song: Song = Song()
        self.song.tracks = [Track('cam.tx', Qt.red), Track('cam.ty', Qt.green), Track('cam.tz', Qt.blue)]

        self.scroll: List[int, int] = [0, 0]
        self.cursorStep: int = 32
        self.renderStep = 16
        self.rowHeight: int = 32
        self.patternNameHeight: int = 22
        self.columnWidth: int = 22
        self.selectionCorners: List[List[int, int], List[int, int]] = [[0, 0], [0, 0]]
        self.variableNameWidth: int = 40
        self._toTopLeft: bool = False

        self.editorDelegate: IcebergSpinBox = IcebergSpinBox()
        self.editorDelegate.setMinimum(-float('inf'))
        self.editorDelegate.setMaximum(float('inf'))
        self.editorDelegate.setWindowFlags(Qt.Popup)
        self.editorDelegate.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.editorDelegate.valueChanged.connect(self._onEdit)
        self.editorDelegate.editingFinished.connect(self._endEdit)
        self.editBiasData: List[float] = []

        self.bias = 1.0
        self.pageSize = 16

        self._drag = False

    def addTrack(self):
        track = Track('New track', QColor(127, 127, 127))
        if self.editTrack(track):
            self.undoStack.push(AddTrack(self.song, self.selectionCorners[1][1] + 1, track, {}))

    def removeTracks(self):
        rowStart, rowEnd = self.selectionCorners[0][1], self.selectionCorners[1][1]
        self.undoStack.push(DeleteTracks(self.song, *range(rowStart, rowEnd + 1)))
        self.undoStack.push(ResetSelection(self))

    def setSelectionTangentMode(self, tangentMode):
        dst = {}
        for track, j in self._iterSelected():
            if j not in track.keys:
                continue
            if tangentMode == TangentMode.Step:
                it = track.keys[j][1]
            else:
                it = tangentMode
            dst.setdefault(track, []).append((j, it, tangentMode))
        self.undoStack.push(SetTangentMode(dst))
        self.setFocus(Qt.MouseFocusReason)

    def setColumnWidth(self, columnWidth):
        self.columnWidth = columnWidth
        self.repaint()

    def setRowHeight(self, rowHeight):
        self.rowHeight = rowHeight
        self.repaint()

    def setBiasStepSize(self, bias):
        self.bias = bias

    def setPageSize(self, pageSize):
        self.pageSize = pageSize

    def setStepFactor(self, stepFactor: int):
        self.setFocus(Qt.MouseFocusReason)  # TODO: Not sure if intuitive
        self.cursorStep = 1 << stepFactor
        self.repaint()

    def _iterSelected(self, exist=False) -> Iterable[Tuple[Track, int]]:
        rowStart, rowEnd = self.selectionCorners[0][1], self.selectionCorners[1][1]
        for i in range(rowStart, rowEnd + 1):
            track = self.song.tracks[i]
            columnStart, columnEnd = self.selectionCorners[0][0], self.selectionCorners[1][0]
            for j in range(columnStart, columnEnd + 1):
                if not exist or j in track.keys:
                    yield track, j

    def _onEdit(self, value: float):
        delta = {}
        for i, (track, j) in enumerate(self._iterSelected()):
            delta.setdefault(track, []).append((j, value + self.editBiasData[i]))
        self.undoStack.push(SetValue(delta))

    def _endEdit(self):
        self.editorDelegate.hide()

    def _beginEdit(self, keyEvent: Optional[QKeyEvent] = None, biasMode: bool = False):
        corner: QPoint = self.mapToGlobal(QPoint(self.variableNameWidth + (self.selectionCorners[0][0] // self.renderStep) * self.columnWidth, self.selectionCorners[0][1] * self.rowHeight))
        geo: QRect = QRect(corner.x(), corner.y() + self.patternNameHeight, self.columnWidth, self.rowHeight)
        self.editorDelegate.setGeometry(geo)
        track: Dict[int, Key] = self.song.tracks[self.selectionCorners[0][1]].keys
        if biasMode:
            value = 0.0
            self.editBiasData = [track.get(j, 0.0) for track, j in self._iterSelected()]
        else:
            value = track.get(self.selectionCorners[0][1], (0.0, TangentMode.Default, TangentMode.Default))
            if isinstance(value, tuple):
                value = value[0]
            self.editBiasData = [0.0 for _ in self._iterSelected()]
        resume = self.editorDelegate.blockSignals(True)
        self.editorDelegate.setValue(value)
        self.editorDelegate.blockSignals(resume)
        self.editorDelegate.show()
        self.editorDelegate.selectAll()
        if keyEvent is not None and keyEvent.key() != Qt.Key_Return:
            self.editorDelegate.keyPressEvent(keyEvent)

    def _selectionToClipboard(self):
        # Copy, we go via strings because I like the idea of pasting to notepad as well
        csv = []
        for i in range(self.selectionCorners[0][1], self.selectionCorners[1][1] + 1):
            track = self.song.tracks[i].keys
            row = []
            for j in range(self.selectionCorners[0][0], self.selectionCorners[1][0] + 1):
                if j in track:
                    row.append(f'{track[j][0]}, {int(track[j][1])}, {int(track[j][2])}')
                else:
                    row.append('-, -, -')
            csv.append(f'{self.song.tracks[i].name}: ' + ', '.join(row))
        QApplication.clipboard().setText('\n'.join(csv))

    def _paste(self):
        # Paste
        rows = QApplication.clipboard().text().split('\n')
        dat: List[List[Optional[Key]]] = []
        for row in rows:
            data = row[row.find(':') + 1:].strip().split(',')
            rowData: List[Optional[Key]] = []
            for val, itm, otm in zip(data[::3], data[1::3], data[2::3]):
                if val.strip() == '-':
                    rowData.append(None)
                else:
                    rowData.append((float(val.strip()), TangentMode(int(itm.strip())), TangentMode(int(otm.strip()))))
            dat.append(rowData)
        untilY = self.selectionCorners[1][1] + 1
        if self.selectionCorners[0][1] == self.selectionCorners[1][1]:
            untilY = self.selectionCorners[0][1] + len(dat)
        delta = {}
        for li, i in enumerate(range(self.selectionCorners[0][1], untilY)):
            if len(dat) != 1 and li >= len(dat):
                break
            rowDat = dat[li] if len(dat) != 1 else dat[0]
            track = self.song.tracks[i]
            untilX = self.selectionCorners[1][0] + 1
            if self.selectionCorners[0][0] == self.selectionCorners[1][0]:
                untilX = self.selectionCorners[0][0] + len(rowDat)
            for lj, j in enumerate(range(self.selectionCorners[0][0], untilX)):
                if len(rowDat) != 1 and lj >= len(rowDat):
                    break
                v = rowDat[lj] if len(rowDat) != 1 else rowDat[0]
                delta.setdefault(track, []).append((j, v))
        self.undoStack.push(SetKey(delta))

    def _arrowMultiSelect(self, key):
        selectionStep = self.renderStep
        # selectionStep = self.cursorStep

        # check if we currently have multi-selection
        single = self.selectionCorners[0] == self.selectionCorners[1]
        if single:
            # if not, determine how things are going to work based on the initial key press
            self._toTopLeft = key in (Qt.Key_Left, Qt.Key_Up)

        # modify multi-selection
        cpy = [self.selectionCorners[0][:], self.selectionCorners[1][:]]
        if key == Qt.Key_Up:
            if self._toTopLeft:
                cpy[0][1] = max(self.selectionCorners[0][1] - 1, 0)
            else:
                cpy[1][1] -= 1
        elif key == Qt.Key_Down:
            if self._toTopLeft:
                cpy[0][1] += 1
            else:
                cpy[1][1] = min(self.selectionCorners[1][1] + 1, len(self.song.tracks) - 1)
        elif key == Qt.Key_Left:
            if self._toTopLeft:
                cpy[0][0] = max(self.selectionCorners[0][0] - selectionStep, 0)
            else:
                cpy[1][0] -= selectionStep
        elif key == Qt.Key_Right:
            if self._toTopLeft:
                cpy[0][0] += selectionStep
            else:
                cpy[1][0] = min(self.selectionCorners[1][0] + selectionStep, self.song.noteCount())
        else:
            return
        self.undoStack.push(SetSelection(self, cpy))

    def _navigation(self, key, alt):
        # Various navigation shortcuts: arrows, page up/down, home/end
        x, y = self.selectionCorners[0]
        if key == Qt.Key_Up:
            y = max(self.selectionCorners[0][1] - 1, 0)
        elif key == Qt.Key_Down:
            y = min(self.selectionCorners[0][1] + 1, len(self.song.tracks) - 1)
        elif key == Qt.Key_Left:
            x = self.snapToCursorStep(max(self.selectionCorners[0][0] - self.cursorStep, 0))
        elif key == Qt.Key_Right:
            x = self.snapToCursorStep(min(self.selectionCorners[0][0] + self.cursorStep, self.song.noteCount()))
        elif key == Qt.Key_PageUp:
            if alt:
                # move to previous pattern start
                bookmarks = sorted(list(self.song.bookmarks.keys()))
                at = bisect.bisect_left(bookmarks, self.selectionCorners[0][0])
                x = bookmarks[max(0, at - 1)]
            else:
                x = self.snapToCursorStep(max(self.selectionCorners[0][0] - self.cursorStep * self.pageSize, 0))
        elif key == Qt.Key_PageDown:
            if alt:
                # move to next pattern start
                bookmarks = sorted(list(self.song.bookmarks.keys()))
                at = bisect.bisect_left(bookmarks, self.selectionCorners[0][0])
                at = min(len(bookmarks) - 1, at)
                if bookmarks[at] == self.selectionCorners[0][0]:
                    at += 1
                    at = min(len(bookmarks) - 1, at)
                x = bookmarks[at]
            else:
                x = self.snapToCursorStep(self.selectionCorners[0][0] + self.cursorStep * self.pageSize)
        elif key == Qt.Key_Home:
            x = 0
        elif key == Qt.Key_End:
            x = self.snapToCursorStep(self.song.noteCount())
        else:
            return

        self.undoStack.push(SetSelection(self, [[x, y], [x, y]]))

    def keyPressEvent(self, event: QKeyEvent):
        # TODO: On some locales period must be replaced with comma
        # Edit as soon as we type a value
        if Qt.Key_0 <= event.key() <= Qt.Key_9 or event.key() in (Qt.Key_Period, Qt.Key_Minus, Qt.Key_Return):
            self._beginEdit(event)
            return

        shift = event.modifiers() & Qt.ShiftModifier == Qt.ShiftModifier
        control = event.modifiers() & Qt.ControlModifier == Qt.ControlModifier
        alt = event.modifiers() & Qt.AltModifier == Qt.AltModifier

        # TODO: Maybe some of these need to be moved to the actual shortcut system
        # Ctrl shortcuts
        if control:
            bias = self.bias
            if shift:
                bias *= 0.1

            if event.key() == Qt.Key_C:  # Copy
                self._selectionToClipboard()
            elif event.key() == Qt.Key_V:  # Paste
                self._paste()
            elif event.key() == Qt.Key_B:  # Bias editor
                self._beginEdit(event, biasMode=True)
            elif event.key() == Qt.Key_Up:  # Bias value
                delta = {}
                for track, j in self._iterSelected(exist=True):
                    delta.setdefault(track, []).append((j, track.keys[j][0] + bias))
                self.undoStack.push(SetValue(delta))
            elif event.key() == Qt.Key_Down:  # Bias value
                delta = {}
                for track, j in self._iterSelected(exist=True):
                    delta.setdefault(track, []).append((j, track.keys[j][0] - bias))
                self.undoStack.push(SetValue(delta))
            return

        # Multi-select
        if event.modifiers() == Qt.ShiftModifier:
            self._arrowMultiSelect(event.key())
            return

        # TODO: Undo/redo
        """# Alt = move track AND cursor
        if alt:
            # TODO: This should probably support multi-select
            if event.key() == Qt.Key_Up:
                if self.selectionCorners[0][1] > 0:
                    self.song.tracks.insert(self.selectionCorners[0][0] - 1, self.song.tracks.pop(self.selectionCorners[0][0]))
                    self.song.data.insert(self.selectionCorners[0][0] - 1, self.song.data.pop(self.selectionCorners[0][0]))
            elif event.key() == Qt.Key_Down:
                if self.selectionCorners[0][1] < len(self.song.tracks) - 1:
                    self.song.tracks.insert(self.selectionCorners[0][0] + 1, self.song.tracks.pop(self.selectionCorners[0][0]))
                    self.song.data.insert(self.selectionCorners[0][0] + 1, self.song.data.pop(self.selectionCorners[0][0]))"""

        # Navigation
        self._navigation(event.key(), alt)

        if event.key() == Qt.Key_K:
            if self.selectionCorners[0][0] in self.song.bookmarks:
                text, accepted = QInputDialog.getText(self, 'Rename pattern', f'Rename "{self.song.bookmarks[self.selectionCorners[0][0]]}"')
                if accepted:
                    self.undoStack.push(SetBookmark(self.song.bookmarks, self.selectionCorners[0][0], text))
            else:
                self.undoStack.push(SetBookmark(self.song.bookmarks, self.selectionCorners[0][0], 'New pattern'))
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            delta = {}
            for track, j in self._iterSelected(exist=True):
                delta.setdefault(track, []).append((j, None))
            self.undoStack.push(SetValue(delta))
        elif event.key() == Qt.Key_E:
            delta = {}
            for track, j in self._iterSelected(exist=True):
                delta.setdefault(track, []).append((j, self.song.evaluateTrack(track, j)))
            self.undoStack.push(SetValue(delta))
        elif event.key() == Qt.Key_I:
            delta = {}
            for track, j in self._iterSelected(exist=True):
                mode = TangentMode((track[j][2] + 1) % 5)
                if mode == TangentMode.Step:
                    delta.setdefault(track, []).append((j, TangentMode.Default, mode))
                else:
                    delta.setdefault(track, []).append((j, mode, mode))
            self.undoStack.push(SetTangentMode(delta))

    def _frameCursor(self):
        # ensure cursor is on the screen by scrolling
        last = ((self.width() - self.variableNameWidth) * self.renderStep) // self.columnWidth + self.snapToCursorStep(self.scroll[0]), self.height() // self.rowHeight + self.scroll[1] - 1
        self.scroll[0] = min(self.scroll[0], self.selectionCorners[0][0])
        self.scroll[1] = min(self.scroll[1], self.selectionCorners[0][1])
        if self.selectionCorners[1][0] > last[0]:
            self.scroll[0] += self.selectionCorners[1][0] - last[0]
            self.scroll[0] = self.snapToVisible(self.scroll[0])
        if self.selectionCorners[1][1] > last[1]:
            self.scroll[1] += self.selectionCorners[1][1] - last[1]

    def mousePressEvent(self, event: QMouseEvent):
        column = ((event.x() - self.variableNameWidth) // self.columnWidth) * self.renderStep + self.scroll[0]
        row = (event.y() - self.patternNameHeight) // self.rowHeight + self.scroll[1]
        if row >= len(self.song.tracks):
            return
        if column < 0 or row < 0:
            return
        if column >= self.song.noteCount():
            return

        self.selectionCorners = [[column, row], [column, row]]
        self._drag = True
        self.repaint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._drag:
            return
        column = ((event.x() - self.variableNameWidth) // self.columnWidth) * self.renderStep + self.scroll[0]
        row = (event.y() - self.patternNameHeight) // self.rowHeight + self.scroll[1]
        if row >= len(self.song.tracks):
            return
        if column < 0 or row < 0:
            return
        if column >= self.song.noteCount():
            return

        self.selectionCorners[0][0] = min(self.selectionCorners[0][0], column)
        self.selectionCorners[1][0] = max(self.selectionCorners[1][0], column)

        self.selectionCorners[0][1] = min(self.selectionCorners[0][1], row)
        self.selectionCorners[1][1] = max(self.selectionCorners[1][1], row)

        self.repaint()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag = False

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.x() < self.variableNameWidth:
            row = (event.y() - self.patternNameHeight) // self.rowHeight + self.scroll[1]
            if 0 <= row < len(self.song.tracks):
                self.editTrack(self.song.tracks[row])

    @staticmethod
    def editTrack(track):
        popup = QDialog()
        popup.setWindowTitle('Edit track')
        popup.setLayout(QVBoxLayout())
        name = QLineEdit()
        name.setText(track.name)
        popup.layout().addWidget(name)
        r = QSpinBox()
        r.setRange(0, 255)
        g = QSpinBox()
        g.setRange(0, 255)
        b = QSpinBox()
        b.setRange(0, 255)
        r.setValue(track.color.red())
        g.setValue(track.color.green())
        b.setValue(track.color.blue())
        popup.layout().addWidget(r)
        popup.layout().addWidget(g)
        popup.layout().addWidget(b)
        ok = QPushButton('ok')
        popup.layout().addWidget(ok)
        ok.clicked.connect(popup.accept)
        name.selectAll()
        name.setFocus()
        popup.exec_()
        if popup.result() != QDialog.Accepted:
            return False
        track.name = name.text()
        track.color = QColor(r.value(), g.value(), b.value())
        return True

    def snapToCursorStep(self, column):
        return floor(column / self.cursorStep) * self.cursorStep

    def snapToVisible(self, column):
        return floor(column / self.renderStep) * self.renderStep

    def paintEvent(self, event):
        CURVE_RENDER_STEP_SIZE = 4

        painter = QPainter(self)
        painter.fillRect(QRect(0, 0, self.width(), self.height()), QColor(20, 20, 20))

        if not self.song.tracks:
            return

        # ensure cursor is on screen
        self._frameCursor()

        for row in range(self.scroll[1], min(len(self.song.tracks), (self.height() - self.patternNameHeight) // self.rowHeight + self.scroll[1] + 1), 1):
            labelGeo = QRect(0, (row - self.scroll[1]) * self.rowHeight + self.patternNameHeight, self.variableNameWidth, self.rowHeight)
            painter.setPen(self.song.tracks[row].color)
            painter.drawText(labelGeo.adjusted(2, 2, 2, 2), Qt.AlignVCenter | Qt.AlignLeft, self.song.tracks[row].name)
            track = self.song.tracks[row].keys
            trackMin, trackMax = None, None
            for value, _, _ in track.values():
                if trackMin is None:
                    trackMin, trackMax = value, value
                    continue
                trackMin = min(trackMin, value)
                trackMax = max(trackMax, value)
            if trackMax == trackMin:
                trackMin, trackMax = -1.0, 1.0
            prevCurvePoint = None
            factor = 1.0 / (trackMin - trackMax) * (self.rowHeight - 4)

            renderColumns = (self.width() - self.variableNameWidth) // self.columnWidth + 1
            for offset in range(renderColumns):
                column = self.scroll[0] + offset * self.renderStep
                valueGeo = QRect(self.variableNameWidth + offset * self.columnWidth, labelGeo.y(), self.columnWidth, labelGeo.height())
                painter.setPen(Qt.white)
                if row == self.scroll[1] and column in self.song.bookmarks:
                    painter.drawText(QRect(valueGeo.x(), 0, 1000, self.patternNameHeight), Qt.AlignVCenter | Qt.AlignLeft, self.song.bookmarks[column])
                if self.selectionCorners[0][0] <= column <= self.selectionCorners[1][0] and self.selectionCorners[0][1] <= row <= self.selectionCorners[1][1]:
                    painter.fillRect(valueGeo, QColor(200, 120, 20))
                elif column % self.song.measureDivisions == 0:
                    painter.fillRect(valueGeo, QColor(80, 80, 80))
                elif column % self.cursorStep == 0:
                    painter.fillRect(valueGeo, QColor(35, 35, 35))
                if column in track:
                    value = f'{track[column][0]:.02f}'
                    painter.setPen(Qt.white)
                else:
                    value = '---'
                    painter.setPen(Qt.black)
                painter.drawText(valueGeo.adjusted(0, 2, 0, 2), Qt.AlignVCenter | Qt.AlignLeft, value)

                painter.setPen(self.song.tracks[row].color)
                painter.setOpacity(0.25)

                n = self.columnWidth
                for x in range(1, n, CURVE_RENDER_STEP_SIZE):
                    t = column + self.renderStep * x / n
                    y = (self.song.evaluateTrack(track, t) - trackMax) * factor + valueGeo.y() + 2
                    curvePoint = QPoint(valueGeo.x() + x, y)
                    if prevCurvePoint is not None:
                        painter.drawLine(prevCurvePoint, curvePoint)
                    prevCurvePoint = curvePoint
                    prevCurvePoint += QPoint(1, 0)
                painter.setOpacity(1.0)


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()

        undoStack = QUndoStack()  # the idea is that you already have an undo stack
        self.setWindowTitle('My awesome demo tool')
        self.setWindowIcon(QIcon('iceberg.png'))

        # you have to make some UI to change the cursor snapping
        tools = QToolBar()

        tangentModeStep = QPushButton(QIcon('step.png'), '')
        tangentModeStep.setToolTip('step')
        tools.addWidget(tangentModeStep)
        tangentModeLinear = QPushButton(QIcon('linear.png'), '')
        tangentModeLinear.setToolTip('linear')
        tools.addWidget(tangentModeLinear)
        tangentModeFlat = QPushButton(QIcon('flat.png'), '')
        tangentModeFlat.setToolTip('flat')
        tools.addWidget(tangentModeFlat)
        tangentModeSpline = QPushButton(QIcon('spline.png'), '')
        tangentModeSpline.setToolTip('spline')
        tools.addWidget(tangentModeSpline)
        tangentModePlateau = QPushButton(QIcon('plateau.png'), '')
        tangentModePlateau.setToolTip('plateau')
        tools.addWidget(tangentModePlateau)

        addTrack = QPushButton('Insert track after cursor')
        tools.addWidget(addTrack)

        removeTracks = QPushButton('Delete tracks at cursor')
        tools.addWidget(removeTracks)

        tools.addWidget(QLabel('Snap: '))
        steps = QComboBox()
        N = 8
        for i in range(N):
            steps.addItem(f'{1 << i}/4')
        tools.addWidget(steps)

        tools.addWidget(QLabel('Bias step size: '))
        bias = QDoubleSpinBox()
        tools.addWidget(bias)

        tools.addWidget(QLabel('Page step size: '))
        pageSize = QSpinBox()
        pageSize.setRange(1, 1000)
        tools.addWidget(pageSize)

        tools.addWidget(QLabel('Cell size: '))
        cellWidth = QSpinBox()
        cellWidth.setRange(1, 1000)
        tools.addWidget(cellWidth)

        cellHeight = QSpinBox()
        cellHeight.setRange(1, 1000)
        tools.addWidget(cellHeight)

        self.addToolBar(tools)

        # the rest of the integration is pretty straight forward
        dock = QDockWidget()
        self.tracker = IcebergTracker(undoStack)
        dock.setWidget(self.tracker)
        dock.setWindowTitle(dock.widget().objectName())
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        steps.currentIndexChanged.connect(lambda x: self.tracker.setStepFactor(N - 1 - x))
        steps.setCurrentIndex(1)  # example default, may want to make this persistent

        bias.valueChanged.connect(self.tracker.setBiasStepSize)
        pageSize.valueChanged.connect(self.tracker.setPageSize)
        cellWidth.valueChanged.connect(self.tracker.setColumnWidth)
        cellHeight.valueChanged.connect(self.tracker.setRowHeight)
        tangentModeStep.clicked.connect(functools.partial(self.tracker.setSelectionTangentMode, TangentMode.Step))
        tangentModeLinear.clicked.connect(functools.partial(self.tracker.setSelectionTangentMode, TangentMode.Linear))
        tangentModeFlat.clicked.connect(functools.partial(self.tracker.setSelectionTangentMode, TangentMode.Flat))
        tangentModeSpline.clicked.connect(functools.partial(self.tracker.setSelectionTangentMode, TangentMode.Spline))
        tangentModePlateau.clicked.connect(functools.partial(self.tracker.setSelectionTangentMode, TangentMode.Plateau))

        addTrack.clicked.connect(self.tracker.addTrack)
        removeTracks.clicked.connect(self.tracker.removeTracks)

        cellWidth.setValue(20)
        cellHeight.setValue(20)


if __name__ == '__main__':
    a = QApplication([])
    with open('dark.qss') as fh:
        a.setStyleSheet(fh.read())
    w = App()
    w.show()
    a.exec_()
