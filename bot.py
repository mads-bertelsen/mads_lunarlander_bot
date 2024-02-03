# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal, Union

import numpy as np

from lunarlander import Instructions

import random

def rotate(current: float, target: float) -> Union[Literal["left", "right"], None]:
    if abs(current - target) < 0.5:
        return
    return "left" if current < target else "right"


def find_landing_site(terrain: np.ndarray, width=40) -> Union[int, None]:
    # Find largest landing site
    n = len(terrain)
    # Find run starts
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(terrain[:-1], terrain[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # Find run lengths
    run_lengths = np.diff(np.append(run_starts, n))

    # Find largest run
    imax = np.argmax(run_lengths)
    start = run_starts[imax]
    end = start + run_lengths[imax]

    # Return location if large enough
    if (end - start) > width:
        loc = int(start + (end - start) * 0.5)
        print("Found landing site at", loc)
        return loc


def good_landing_site(terrain, target_site, width):
    value = terrain[target_site]
    for index in range(int(target_site- 0.5*width), int(target_site + 0.5*width)):
        if terrain[index] != value:
            return False

    return True


def angle_for_go_to_x(target_x, x, y, vx, g, MAX_ANGLE=40, MAX_VX=30, verbose=False):

    estimate_ax = 2 * np.tan(MAX_ANGLE * np.pi / 180) * g / 3

    accelerate_x = False
    if target_x < x:
        # move left
        move_direction = -1
        accelerate_angle = 1
    else:
        # move right
        move_direction = 1
        accelerate_angle = -1

    if move_direction*vx < 0:
        # moving in the wrong direction
        if verbose:
            print("pos:", np.array([x, y]), "target_x:", np.array(target_x), "vx:", np.array([vx]), "WRONG WAY")
        return MAX_ANGLE * accelerate_angle

    if MAX_VX < abs(vx):
        if verbose:
            print("pos:", np.array([x, y]), "target_x:", np.array(target_x), "vx:", np.array([vx]), "TOO FAST")
        return -MAX_ANGLE * accelerate_angle

    vx_limit = estimate_ax * (np.sqrt(2 * abs(target_x - x) / estimate_ax) - 0.7)
    if vx_limit < 0:
        vx_limit = 0

    if vx_limit < abs(vx):
        if verbose:
            print("pos:", np.array([x, y]), "target_x:", np.array(target_x), "vx:", np.array([vx]), "SLOW TO TARGET, vlim = ", vx_limit)
        return -MAX_ANGLE * accelerate_angle

    if 0.8*vx_limit < abs(vx):
        if verbose:
            print("pos:", np.array([x, y]), "target_x:", np.array(target_x), "vx:", np.array([vx]), "Coasting towards target, vlim=", vx_limit)
        return 0

    if 0.9*MAX_VX < vx:
        if verbose:
            print("pos:", np.array([x, y]), "target_x:", np.array(target_x), "vx:", np.array([vx]), "Coasting towards target, (max) vlim=", vx_limit)
        return 0

    # If all good, go faster towards target
    if verbose:
        print("pos:", np.array([x, y]), "target_x:", np.array(target_x), "vx:", np.array([vx]), "Towards target, vlim=", vx_limit)
    return MAX_ANGLE * accelerate_angle


class Bot:
    """
    This is the lander-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "Mads"  # This is your team name
        self.avatar = 0  # Optional attribute
        self.flag = "dk"  # Optional attribute
        self.initial_manoeuvre = True
        self.target_site = None
        self.g = 1.62
        self.old_vx = None
        self.started_landing = False

        original_options = np.get_printoptions()
        # Set the new print options to limit to 2 significant digits
        np.set_printoptions(precision=2, suppress=True)

    def run(
        self,
        t: float,
        dt: float,
        terrain: np.ndarray,
        players: dict,
        asteroids: list,
    ):
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in seconds.
        dt:
            The time step in seconds.
        terrain:
            The (1d) array representing the lunar surface altitude.
        players:
            A dictionary of the players in the game. The keys are the team names and
            the values are the information about the players.
        asteroids:
            A list of the asteroids currently flying.
        """
        instructions = Instructions()

        verbose = True

        me = players[self.team]
        x, y = me.position
        vx, vy = me.velocity
        head = me.heading

        terrain_max = 25 + max(terrain)

        # Perform an initial rotation to get the LEM pointing upwards
        if self.initial_manoeuvre:
            if y < terrain_max:
                instructions.main = True
                command = rotate(current=head, target=0)
                if command == "left":
                    instructions.left = True
                elif command == "right":
                    instructions.right = True

                return instructions

            if vx > 12:
                instructions.main = True
            else:
                command = rotate(current=head, target=0)
                if command == "left":
                    instructions.left = True
                elif command == "right":
                    instructions.right = True
                else:
                    self.initial_manoeuvre = False
            return instructions

        landing_site_size = 40

        if me.fuel < 350:
            landing_site_size = 36

        if me.fuel < 340:
            landing_site_size = 32

        # Search for a suitable landing site
        if self.target_site is None:
            self.target_site = find_landing_site(terrain, width=landing_site_size)
        else:
            if not good_landing_site(terrain, self.target_site, 28):
                self.started_landing = False
                self.target_site = None

        if self.target_site is None:
            # get to target hover
            if vy < 0:
                instructions.main = True

            if y < terrain_max and vy < 8:
                instructions.main = True

        else:
            # Move to the target
            if self.started_landing:
                location_limit = 150
            else:
                location_limit = 3

            if abs(self.target_site - x) > location_limit:

                if verbose:
                    print("Goto X mode")
                if vy < 0:
                    instructions.main = True

                if y < terrain_max and vy < 8:
                    instructions.main = True

                if self.started_landing:
                    used_maximum_angle = 25
                else:
                    used_maximum_angle = 40

                desired_angle = angle_for_go_to_x(self.target_site, x, y, vx, self.g, MAX_ANGLE=used_maximum_angle)

                command = rotate(current=head, target=desired_angle)
                if command == "left":
                    instructions.left = True
                elif command == "right":
                    instructions.right = True

                if self.old_vx is not None:
                    acc = (vx - self.old_vx)/dt

                    if abs(acc) > 0.1:
                        pass
                        #print("x acceleration = ",  acc)

                self.old_vx = vx

                return instructions

            else:
                # Decend
                self.started_landing = True

                height_diff = y - terrain[self.target_site]
                position_error = abs(x - self.target_site)

                position_diff = self.target_site - x

                angle_factor = 1
                direction_sign = 0
                if self.target_site > x and vx > 0:
                    # Am to the left, going right
                    # Want to continue

                    direction_sign = 0

                    if position_error > 3:
                        if abs(vx) > 0.7:
                            direction_sign = -1

                    else:
                        if abs(vx) > 0.7:
                            angle_factor = 0.7
                            direction_sign = 1

                    if position_error < 1:
                        angle_factor = 0.5

                    if verbose:
                        print("1 t:", self.target_site, "x:", x, "vx:", vx, "direction_sign", angle_factor*direction_sign)

                elif self.target_site < x and vx > 0:
                    # Am to the right, going right
                    # Go left

                    direction_sign = 1

                    if position_error < 3 and abs(vx) < 1:
                        angle_factor = 0.5

                    if verbose:
                        print("2 t:", self.target_site, "x:", x, "vx:", vx, "direction_sign", direction_sign)

                elif self.target_site > x and vx < 0:
                    # Am to the left, going left
                    # Go right

                    direction_sign = -1

                    if position_error < 3 and abs(vx) < 1:
                        angle_factor = 0.5

                    if verbose:
                        print("3 t:", self.target_site, "x:", x, "vx:", vx, "direction_sign", angle_factor*direction_sign)

                elif self.target_site < x and vx < 0:
                    # Am to the right, going left
                    # Want to continue

                    direction_sign = 0

                    if position_error > 3:
                        if abs(vx) > 0.7:
                            direction_sign = 1

                    else:
                        if abs(vx) > 0.7:
                            angle_factor = 0.7
                            direction_sign = -1

                    if position_error < 2:
                        angle_factor = 0.5

                    if verbose:
                        print("4 t:", self.target_site, "x:", x, "vx:", vx, "direction_sign", angle_factor*direction_sign)

                if height_diff > 100:
                    if angle_factor < 1:
                        maximum_angle = min([position_error, 12])
                    else:
                        maximum_angle = min([position_error, 20])
                else:
                    maximum_angle = min([position_error, 4])

                command = rotate(current=head, target=angle_factor*direction_sign*maximum_angle)
                if command == "left":
                    instructions.left = True
                elif command == "right":
                    instructions.right = True


                """
                maximum_angle = min([3 * position_error, 10])

                if height_diff < 100:
                    maximum_angle = min([3 * position_error, 4])

                desired_angle = angle_for_go_to_x(self.target_site, x, y, vx, self.g, MAX_ANGLE=maximum_angle)
                command = rotate(current=head, target=desired_angle)
                if command == "left":
                    instructions.left = True
                elif command == "right":
                    instructions.right = True
                
                """

                SAFETY_FACTOR = 0.8
                estimated_y_acc = SAFETY_FACTOR * 3.0 * self.g * np.sin(maximum_angle * np.pi / 180)
                vy_limit = estimated_y_acc * np.sqrt(2*height_diff/estimated_y_acc)

                if vy_limit < 5:
                    vy_limit = 4.5

                if vy < -vy_limit:
                    instructions.main = True
                else:
                    instructions.main = False

                if position_error > 3:
                    if abs(height_diff) > 60:
                        pass

                if random.uniform(0, 1) > 0.85:
                    instructions.main = True

                if position_error > 10:
                    if vy < 0:
                        instructions.main = True

                """
                print("Descend mode x = ", np.array([x]), "y = ", np.array([y]), " vy = ", np.array([vy]),
                      "vy_lim =", np.array([vy_limit]), "max_angle=", np.array(maximum_angle))
                """

                return instructions



        return instructions
