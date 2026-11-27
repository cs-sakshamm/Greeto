"""
Programmatic face rendering for the Greeto digital character.

Every facial element is drawn with QPainter primitives - paths, gradients
and strokes. This module never loads an image, SVG, GIF or video: the face
is geometry, not a pre-rendered asset.

The renderer is stateless. It receives a parameter object and paints one
frame from it, which keeps animation and drawing completely separate.
"""

from __future__ import annotations

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)


def _clamp(value, low=0.0, high=1.0):
    """Clamp ``value`` into the inclusive range ``low``..``high``."""

    return low if value < low else high if value > high else value


def _alpha(color, alpha):
    """Return a copy of ``color`` with a new alpha channel."""

    result = QColor(color)
    result.setAlpha(int(_clamp(alpha, 0, 255)))
    return result


def _mix(first, second, amount):
    """Linear blend between two colours. ``amount`` 0 = first, 1 = second."""

    amount = _clamp(amount)

    return QColor(
        int(first.red() + (second.red() - first.red()) * amount),
        int(first.green() + (second.green() - first.green()) * amount),
        int(first.blue() + (second.blue() - first.blue()) * amount),
        int(first.alpha() + (second.alpha() - first.alpha()) * amount),
    )


class FacePalette:
    """
    The single colour identity of the character.

    One palette is reused for every expression so the face stays
    recognisable; expressions change geometry and intensity, not identity.
    """

    ACCENT = QColor(56, 224, 255)
    ACCENT_INK = QColor(12, 86, 112)
    WARM = QColor(255, 198, 108)

    EYE_TOP = QColor(7, 24, 36)
    EYE_BOTTOM = QColor(20, 88, 112)
    PUPIL_CORE = QColor(238, 253, 255)

    MOUTH_INNER = QColor(4, 14, 22)
    MOUTH_DEEP = QColor(9, 40, 54)

    PANEL_TOP = QColor(20, 40, 58, 118)
    PANEL_MID = QColor(11, 24, 36, 150)
    PANEL_BOTTOM = QColor(6, 12, 20, 178)


class FaceRenderer:
    """
    Draws the face.

    All geometry is expressed as a fraction of ``side`` - the smaller
    widget dimension - so the face keeps identical proportions at every
    resolution.
    """

    # Layout, as fractions of the drawing box.
    PANEL_W = 0.80
    PANEL_H = 0.70
    PANEL_RADIUS = 0.155

    EYE_DX = 0.175
    EYE_DY = -0.045
    EYE_HALF_W = 0.100
    EYE_HALF_H = 0.054
    PUPIL_R = 0.029

    BROW_DY = -0.150
    BROW_HALF_W = 0.086

    MOUTH_DY = 0.150
    MOUTH_HALF_W = 0.112

    # ------------------------------------------------------------------

    def paint(self, painter, rect, params):
        """Paint one frame of the face into ``rect``."""

        side = min(rect.width(), rect.height())

        if side < 8.0:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        centre_x = rect.center().x()
        # A barely visible breathing bob. It is what stops the face from
        # looking frozen when nobody is interacting with it.
        centre_y = rect.center().y() + params.breath * side * 0.004

        accent = _mix(FacePalette.ACCENT, FacePalette.WARM, _clamp(params.warmth))

        self._draw_aura(painter, centre_x, centre_y, side, params, accent)
        self._draw_panel(painter, centre_x, centre_y, side, params, accent)

        self._draw_brow(painter, centre_x, centre_y, side, params, accent, -1)
        self._draw_brow(painter, centre_x, centre_y, side, params, accent, 1)

        self._draw_eye(painter, centre_x, centre_y, side, params, accent, -1)
        self._draw_eye(painter, centre_x, centre_y, side, params, accent, 1)

        self._draw_mouth(painter, centre_x, centre_y, side, params, accent)

        painter.restore()

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------

    def _draw_aura(self, painter, cx, cy, side, params, accent):
        """Soft halo behind the panel - cheap radial gradient, no blur."""

        # Kept fully inside the widget: a halo cut off by the widget edge
        # can produce a visible seam on some displays.
        radius = side * 0.48
        centre = QPointF(cx, cy - side * 0.02)

        gradient = QRadialGradient(centre, radius)
        gradient.setColorAt(0.0, _alpha(accent, 14 + 26 * params.glow))
        gradient.setColorAt(0.62, _alpha(accent, 5 + 10 * params.glow))
        gradient.setColorAt(1.0, _alpha(accent, 0))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(centre, radius, radius * 0.88)

    # ------------------------------------------------------------------
    # Facial structure
    # ------------------------------------------------------------------

    def _draw_panel(self, painter, cx, cy, side, params, accent):
        """The face plate the features live on."""

        width = side * self.PANEL_W
        height = side * self.PANEL_H

        rect = QRectF(
            cx - width * 0.5,
            cy - height * 0.5,
            width,
            height,
        )

        radius = side * self.PANEL_RADIUS

        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)

        fill = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        fill.setColorAt(0.0, FacePalette.PANEL_TOP)
        fill.setColorAt(0.55, FacePalette.PANEL_MID)
        fill.setColorAt(1.0, FacePalette.PANEL_BOTTOM)

        painter.setBrush(QBrush(fill))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(
            QPen(
                _alpha(accent, 18 + 26 * params.glow),
                max(1.0, side * 0.0026),
            )
        )
        painter.drawPath(path)

        # Glass highlight along the top edge.
        highlight = QLinearGradient(rect.left(), 0.0, rect.right(), 0.0)
        highlight.setColorAt(0.0, _alpha(accent, 0))
        highlight.setColorAt(0.5, _alpha(accent, 40 + 24 * params.glow))
        highlight.setColorAt(1.0, _alpha(accent, 0))

        painter.setPen(QPen(QBrush(highlight), max(1.0, side * 0.0022)))

        line_y = rect.top() + side * 0.030
        inset = side * 0.13

        painter.drawLine(
            QPointF(rect.left() + inset, line_y),
            QPointF(rect.right() - inset, line_y),
        )

    # ------------------------------------------------------------------
    # Eyes
    # ------------------------------------------------------------------

    def _draw_eye(self, painter, cx, cy, side, params, accent, direction):
        """
        Draw one eye.

        ``direction`` is -1 for the left eye and +1 for the right eye.
        The eye outline is a closed cubic almond; its height is driven by
        the openness parameter, which is how blinking is rendered.
        """

        ex = cx + direction * side * self.EYE_DX
        ey = cy + side * self.EYE_DY

        half_w = side * self.EYE_HALF_W

        openness = params.eye_open_l if direction < 0 else params.eye_open_r
        openness = _clamp(openness)

        half_h = max(side * self.EYE_HALF_H * openness, side * 0.0035)

        upper = 1.30 * half_h

        # A positive eye_smile lifts the lower lid, producing the narrow
        # curved eye that reads as a genuine smile.
        lower = (1.46 - 1.95 * _clamp(params.eye_smile, -1.0, 1.0)) * half_h
        lower = max(lower, 0.50 * half_h)

        outline = QPainterPath(QPointF(ex - half_w, ey))
        outline.cubicTo(
            QPointF(ex - half_w * 0.55, ey - upper),
            QPointF(ex + half_w * 0.55, ey - upper),
            QPointF(ex + half_w, ey),
        )
        outline.cubicTo(
            QPointF(ex + half_w * 0.55, ey + lower),
            QPointF(ex - half_w * 0.55, ey + lower),
            QPointF(ex - half_w, ey),
        )

        glow = _clamp(params.glow)

        # Halo. Drawn as a wide soft stroke around the eye shape rather
        # than a circular bloom, which would read as a disc behind the eye.
        painter.setBrush(Qt.NoBrush)

        painter.setPen(
            QPen(
                _alpha(accent, 24 + 18 * glow),
                max(4.0, side * 0.030),
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
        )
        painter.drawPath(outline)

        painter.setPen(
            QPen(
                _alpha(accent, 20 + 14 * glow),
                max(2.0, side * 0.013),
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
        )
        painter.drawPath(outline)

        # Eye body: a dark lens lit from below, so it reads as a screen.
        body = QLinearGradient(ex, ey - half_h, ex, ey + half_h)
        body.setColorAt(0.0, FacePalette.EYE_TOP)
        body.setColorAt(0.55, _mix(FacePalette.EYE_TOP, FacePalette.EYE_BOTTOM, 0.55))
        body.setColorAt(1.0, _mix(FacePalette.EYE_BOTTOM, accent, 0.55))

        painter.setBrush(QBrush(body))
        painter.setPen(
            QPen(
                _alpha(accent, 150 + 70 * glow),
                max(1.0, side * 0.0032),
            )
        )
        painter.drawPath(outline)

        painter.save()
        painter.setClipPath(outline)

        # Light pooling in the lower half of the eye.
        inner = QRadialGradient(QPointF(ex, ey + half_h * 1.25), half_w * 1.30)
        inner.setColorAt(0.0, _alpha(accent, 105 + 40 * glow))
        inner.setColorAt(0.55, _alpha(accent, 34))
        inner.setColorAt(1.0, _alpha(accent, 0))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(inner))
        painter.drawEllipse(
            QPointF(ex, ey + half_h * 1.25),
            half_w * 1.30,
            half_h * 2.6,
        )

        # Pupil: a luminous focus point, not a drawn eyeball with an
        # iris ring and a specular highlight.
        pupil_r = side * self.PUPIL_R

        px = ex + params.pupil_x * half_w * 0.52
        py = ey + params.pupil_y * half_h * 0.38

        halo_radius = pupil_r * 2.35
        halo = QRadialGradient(QPointF(px, py), halo_radius)
        halo.setColorAt(0.0, _alpha(accent, 105 + 45 * glow))
        halo.setColorAt(0.42, _alpha(accent, 42))
        halo.setColorAt(1.0, _alpha(accent, 0))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(halo))
        painter.drawEllipse(QPointF(px, py), halo_radius, halo_radius)

        core = QRadialGradient(QPointF(px, py), pupil_r)
        core.setColorAt(0.0, _alpha(FacePalette.PUPIL_CORE, 245))
        core.setColorAt(0.32, _mix(FacePalette.PUPIL_CORE, accent, 0.45))
        core.setColorAt(0.72, _mix(accent, FacePalette.ACCENT_INK, 0.35))
        core.setColorAt(1.0, _alpha(accent, 30))

        painter.setBrush(QBrush(core))
        painter.drawEllipse(QPointF(px, py), pupil_r, pupil_r)

        painter.restore()

        # A closed eye is a lid line, not an empty socket.
        if openness < 0.14:
            strength = 1.0 - (openness / 0.14)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(
                QPen(
                    _alpha(accent, 90 + 110 * strength),
                    max(1.0, side * 0.0040),
                    Qt.SolidLine,
                    Qt.RoundCap,
                )
            )
            painter.drawLine(
                QPointF(ex - half_w * 0.86, ey),
                QPointF(ex + half_w * 0.86, ey),
            )

    # ------------------------------------------------------------------
    # Eyebrows
    # ------------------------------------------------------------------

    def _draw_brow(self, painter, cx, cy, side, params, accent, direction):
        """One eyebrow. Raise moves it up, tilt lifts the inner end."""

        if direction < 0:
            raise_amount = params.brow_raise_l
            tilt = params.brow_tilt_l
        else:
            raise_amount = params.brow_raise_r
            tilt = params.brow_tilt_r

        browser_x = cx + direction * side * self.EYE_DX

        base_y = (
            cy
            + side * self.BROW_DY
            - _clamp(raise_amount, -1.0, 1.0) * side * 0.030
        )

        half_w = side * self.BROW_HALF_W

        inner_x = browser_x - direction * half_w
        outer_x = browser_x + direction * half_w

        inner_y = base_y - tilt * side * 0.020
        outer_y = base_y + tilt * side * 0.006

        arch = side * 0.024 + _clamp(raise_amount, -1.0, 1.0) * side * 0.006

        path = QPainterPath(QPointF(outer_x, outer_y))
        path.cubicTo(
            QPointF(outer_x - direction * half_w * 0.55, outer_y - arch),
            QPointF(inner_x + direction * half_w * 0.45, inner_y - arch * 0.92),
            QPointF(inner_x, inner_y),
        )

        width = max(1.5, side * 0.0072)
        glow = _clamp(params.glow)

        painter.setBrush(Qt.NoBrush)

        # Soft under-stroke instead of an expensive blur effect.
        painter.setPen(
            QPen(
                _alpha(accent, 34 + 26 * glow),
                width * 2.6,
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
        )
        painter.drawPath(path)

        painter.setPen(
            QPen(
                _alpha(_mix(accent, QColor(255, 255, 255), 0.30), 175 + 55 * glow),
                width,
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
        )
        painter.drawPath(path)

    # ------------------------------------------------------------------
    # Mouth
    # ------------------------------------------------------------------

    def _draw_mouth(self, painter, cx, cy, side, params, accent):
        """
        Draw the mouth.

        Two symmetric cubic segments describe the upper and lower lip. The
        shape collapses to a curve when closed, which is why the lip line
        is stroked separately from the filled interior.
        """

        base_y = cy + side * self.MOUTH_DY

        half_w = side * self.MOUTH_HALF_W * _clamp(params.mouth_width, 0.5, 1.6)

        opening = _clamp(params.mouth_open)
        half_open = opening * side * 0.052

        depth = params.mouth_curve * side * 0.052

        upper_mid = base_y - half_open * 0.85 + depth * 0.30
        lower_mid = base_y + half_open + depth * 1.00

        upper_ctrl = (upper_mid - 0.25 * base_y) / 0.75
        lower_ctrl = (lower_mid - 0.25 * base_y) / 0.75

        shape = QPainterPath(QPointF(cx - half_w, base_y))
        shape.cubicTo(
            QPointF(cx - half_w * 0.55, upper_ctrl),
            QPointF(cx + half_w * 0.55, upper_ctrl),
            QPointF(cx + half_w, base_y),
        )
        shape.cubicTo(
            QPointF(cx + half_w * 0.55, lower_ctrl),
            QPointF(cx - half_w * 0.55, lower_ctrl),
            QPointF(cx - half_w, base_y),
        )
        shape.closeSubpath()

        thickness = lower_mid - upper_mid
        glow = _clamp(params.glow)

        if thickness > 1.2:

            interior = QLinearGradient(cx, upper_mid, cx, lower_mid)
            interior.setColorAt(0.0, FacePalette.MOUTH_INNER)
            interior.setColorAt(
                1.0,
                _mix(FacePalette.MOUTH_DEEP, accent, 0.20),
            )

            painter.setBrush(QBrush(interior))
            painter.setPen(
                QPen(
                    _alpha(accent, 120 + 60 * glow),
                    max(1.0, side * 0.0026),
                )
            )
            painter.drawPath(shape)

        # Lip line. When the mouth is open this would cut across the
        # interior, so it is only drawn for near-closed shapes.
        if thickness <= side * 0.020:

            lip_mid = base_y + depth * 0.62 - half_open * 0.06
            lip_ctrl = (lip_mid - 0.25 * base_y) / 0.75

            lip = QPainterPath(QPointF(cx - half_w, base_y))
            lip.cubicTo(
                QPointF(cx - half_w * 0.55, lip_ctrl),
                QPointF(cx + half_w * 0.55, lip_ctrl),
                QPointF(cx + half_w, base_y),
            )

            painter.setBrush(Qt.NoBrush)

            painter.setPen(
                QPen(
                    _alpha(accent, 40 + 20 * glow),
                    max(2.0, side * 0.016),
                    Qt.SolidLine,
                    Qt.RoundCap,
                )
            )
            painter.drawPath(lip)

            painter.setPen(
                QPen(
                    _alpha(
                        _mix(accent, QColor(255, 255, 255), 0.38),
                        170 + 60 * glow,
                    ),
                    max(1.4, side * 0.0055),
                    Qt.SolidLine,
                    Qt.RoundCap,
                )
            )
            painter.drawPath(lip)
