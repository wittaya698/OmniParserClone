"""
Adapted from https://github.com/google-research/google-research/tree/master/android_in_the_wild
"""

import jax
import jax.numpy as jnp
import numpy as np

# import action_type as action_type_lib
import enum


class ActionType(enum.IntEnum):
    # Placeholders for unused enum values
    UNUSED_0 = 0
    UNUSED_1 = 1
    UNUSED_2 = 2
    UNUSED_8 = 8
    UNUSED_9 = 9

    ########### Agent actions ###########

    # A type action that sends text to the emulator. Note that this simply sends
    # text and does not perform any clicks for element focus or enter presses for
    # submitting text.
    TYPE = 3

    # The dual point action used to represent all gestures.
    DUAL_POINT = 4

    # These actions differentiate pressing the home and back button from touches.
    # They represent explicit presses of back and home performed using ADB.
    PRESS_BACK = 5
    PRESS_HOME = 6

    # An action representing that ADB command for hitting enter was performed.
    PRESS_ENTER = 7

    ########### Episode status actions ###########

    # An action used to indicate the desired task has been completed and resets
    # the environment. This action should also be used in the case that the task
    # has already been completed and there is nothing to do.
    # e.g. The task is to turn on the Wi-Fi when it is already on
    STATUS_TASK_COMPLETE = 10

    # An action used to indicate that desired task is impossible to complete and
    # resets the environment. This can be a result of many different things
    # including UI changes, Android version differences, etc.
    STATUS_TASK_IMPOSSIBLE = 11


_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4

# Interval determining if an action is a tap or a swipe.
_SWIPE_DISTANCE_THRESHOLD = 0.04


def _yx_in_bounding_boxes(yx, bounding_boxes):
    raise NotImplementedError("Function _yx_in_bounding_boxes is not implemented")


def _resize_annotation_bounding_boxes(
    annotation_positions,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
):
    raise NotImplementedError(
        "Function _resize_annotation_bounding_boxes is not implemented"
    )


def is_tap_action(normalized_start_yx, normalized_end_yx):
    raise NotImplementedError("Function is_tap_action is not implemented")


def _is_non_dual_point_action(action_type):
    raise NotImplementedError("Function _is_non_dual_point_action is not implemented")


def _check_tap_actions_match(
    tap_1_yx,
    tap_2_yx,
    annotation_positions,
    matching_tap_distance_threshold_screen_percentage,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
):
    raise NotImplementedError("Function _check_tap_actions_match is not implemented")


def _check_drag_actions_match(
    drag_1_touch_yx,
    drag_1_lift_yx,
    drag_2_touch_yx,
    drag_2_lift_yx,
):
    raise NotImplementedError("Function _check_drag_actions_match is not implemented")


def check_actions_match(
    action_1_touch_yx,
    action_1_lift_yx,
    action_1_action_type,
    action_2_touch_yx,
    action_2_lift_yx,
    action_2_action_type,
    annotation_positions,
    tap_distance_threshold=_TAP_DISTANCE_THRESHOLD,
    annotation_width_augment_fraction=ANNOTATION_WIDTH_AUGMENT_FRACTION,
    annotation_height_augment_fraction=ANNOTATION_HEIGHT_AUGMENT_FRACTION,
):
    raise NotImplementedError("Function check_actions_match is not implemented")


def action_2_format(step_data):
    raise NotImplementedError("Function action_2_format is not implemented")


def pred_2_format(step_data):
    raise NotImplementedError("Function pred_2_format is not implemented")


def pred_2_format_simplified(step_data):
    raise NotImplementedError("Function pred_2_format_simplified is not implemented")
