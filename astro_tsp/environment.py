import os
import re

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

PRIORITY_GRANULATION = 1  # Round to n decimal places for priority-related ops
DISTANCE_GRANULATION = 2  # Round to n decimal places for distance-related ops
TRAVEL_SPEED = 1.4  # Arbitrary distance/seconds speed


class EnvironmentPayload:
    """Deals with setting up the input data for the Environment"""

    def __init__(
                self,
                *,
                n_targets: int | None = None,
                min_x: int | None = None,
                max_x: int | None = None,
                min_y: int | None = None,
                max_y: int | None = None,
                min_gen_prio: int | None = None,
                max_gen_prio: int | None = None,
                data_directory: str = ""
            ) -> None:
        """
        Initializes the payload for a delivery environment

        Some constraints that are checked by this class (and some notes);
        1) If data_directory is provided, n_targets and (min|max)_(x|y) cannot
        be provided
        2) If a data_directory is provided (x = RA, y = DECL):
            - min_x: 0
            - max_x: +24
            - min_y: -90
            - max_y: +90
        3) max_x > min_x and max_y > min_y
        4) max_gen_prio is larger than min_gen_prio, and is at least 1
        5) Regardless of the input parameters, a field `max_distance` will be
        created, which is the maximum possible distance based on the dimensions
        provided
        6) priorities will also be calculated, along with max_priority and
        total_priority
        7) distances will also be calculated between all generated points
        8) There must be at least one target
        9) If a data directory was provided, self.time_windows will be avail.

        Parameters
        ----------
        n_targets : int
            The number of targets to create
        min_x : int
            The minimum value in the X axis
        max_x : int
            The maximum value in the X axis
        min_y : int
            The minimum value in the Y axis
        max_y : int
            The maximum value in the Y axis
        min_gen_prio : int
            The minimum priority that could be generated for a target
        max_gen_prio : int
            The maximum priority that could be generated for a target
        data_directory : str
            The directory in which to find AstroTSP data, if we're loading via
            that
        """

        if data_directory:
            assert n_targets is None
            assert min_x is None and max_x is None
            assert min_y is None and max_y is None
            self.min_x, self.max_x = 0, 24
            self.min_y, self.max_y = -90, 90
        else:
            assert n_targets
            assert min_x is not None and max_x is not None
            assert min_y is not None and max_y is not None
            self.n_targets = n_targets
            self.min_x, self.max_x = min_x, max_x
            self.min_y, self.max_y = min_y, max_y

        assert min_gen_prio and max_gen_prio
        self.min_gen_prio, self.max_gen_prio = min_gen_prio, max_gen_prio

        assert self.max_x > self.min_x and self.max_y > self.min_y
        assert self.max_gen_prio > self.min_gen_prio > 0

        self._generate_targets(data_directory)

        self.n_targets = len(self.all_coords)
        assert self.n_targets > 0
        if data_directory:
            assert self.time_windows is not None
            assert self.earliest_start != -1

        self._generate_priorities()
        self._calculate_distances()

        self.max_distance = np.sqrt(
            (self.max_x - self.min_x) ** 2
            +
            (self.max_y - self.min_y) ** 2
        )

    def _generate_targets(self, data_directory: str = "") -> None:
        """
        Generate Targets for the Environment
        """

        if not data_directory:
            self.all_coords = EnvironmentPayload.get_coords(
                self.n_targets,
                self.min_x,
                self.max_x,
                self.min_y,
                self.max_y,
            )
            self.time_windows = None
            self.earliest_start = None
            self.exptimes = None  # Initialize exptimes as None for generated data
            return

        # Handling Astro-TSP data
        (
            self.all_coords,
            self.time_windows,
            earliest_start,
            self.exptimes,  # Store exptimes data
        ) = EnvironmentPayload.get_astro_data(data_directory)
        
        self.earliest_start = earliest_start
        self.all_x = self.all_coords[:, 0]
        self.all_y = self.all_coords[:, 1]

    def _generate_priorities(self) -> None:
        self.priorities = [
            np.random.randint(self.min_gen_prio, self.max_gen_prio + 1)
            for _ in range(self.n_targets)
        ]
        self.max_priority = max(self.priorities)
        self.total_priority = sum(self.priorities)

    def _calculate_distances(self) -> None:
        """
        Returns the distance between each of the two points in all_x and all_y
        """
        all_coords = np.column_stack([self.all_x, self.all_y])
        self.distance_matrix = cdist(all_coords, all_coords)

    @staticmethod
    def get_coords(
                n_targets: int,
                min_x: int,
                max_x: int,
                min_y: int,
                max_y: int
            ) -> NDArray[np.floating]:
        """Create random coordinates using Numpy"""
        coords = np.zeros((n_targets, 2), dtype=np.float64)

        for i in range(n_targets):
            while True:
                new_point = np.array([
                    np.random.uniform(min_x, max_x),
                    np.random.uniform(min_y, max_y)
                ])
                if EnvironmentPayload.distance_check(new_point, coords[:i]):
                    coords[i] = new_point
                    break

        return coords

    @staticmethod
    def get_astro_data(
                data_directory: str
            ) -> tuple[
                    NDArray[np.floating],
                    list[list[tuple[float, float]]],
                    tuple[float, int],
                    list[list[tuple[float, float]]]
                ]:
        """
        Retrieve all data from the provided Astro-TSP data directory
        """
        found_coords = set()
        coords = []
        time_windows: list[list[tuple[float, float]]] = []
        earliest_start: tuple[float, int] = (-1, -1)
        exptimes: list[list[tuple[float, float]]] = []
        for _filename in os.listdir(data_directory):
            if not _filename.endswith(".csv"):
                print(f"Skipping file {_filename}: Not a CSV.")
                continue

            filename = os.path.join(data_directory, _filename)
            file = open(filename, "r")
            file.readline()  # Discard blank line

            line = file.readline()
            while line:
                all_data = line.split()
                curr_time_window: list[tuple[float, float]] = []
                curr_exptime: list[tuple[float, float]] = []
                if len(all_data) == 1:
                    file.readline()  # Discard heading line
                    line = file.readline()  # The first line of data
                    all_data = line.split()

                    # Make sure we aren't adding the same coordinate twice
                    point = (float(all_data[2]), float(all_data[3]))
                    if point in found_coords:
                        continue
                    if not EnvironmentPayload.distance_check(point, coords):
                        continue

                    coords.append(point)
                    found_coords.add(point)

                    start_time = EnvironmentPayload.date_string_to_seconds(
                        all_data[4]
                    )
                    if start_time < earliest_start[0] or \
                            earliest_start[0] == -1:
                        earliest_start = (start_time, len(coords))

                    while len(all_data) > 5:
                        start_time = EnvironmentPayload.date_string_to_seconds(
                            all_data[4]
                        )
                        curr_time_window.append(
                            (start_time, float(all_data[6]))
                        )
                        
                        # Store exptime data (index 5 contains OTMexptime)
                        if len(all_data) > 5:
                            curr_exptime.append(
                                (start_time, float(all_data[5]))
                            )

                        line = file.readline()
                        all_data = line.split()

                    time_windows.append(curr_time_window)
                    exptimes.append(curr_exptime)
                    curr_time_window = []
                    curr_exptime = []

                line = file.readline()

            file.close()

        return (
            np.array(coords, dtype=np.float64),
            time_windows,
            earliest_start,
            exptimes
        )

    @staticmethod
    def date_string_to_seconds(date_string: str) -> float:
        """
        Given a String in the format YYYY-MM-DDTHH:MM:SS, this method
        calculates the time in seconds

        Parameters
        ----------
        date_string : str
            The formatted String to calculate from

        Returns
        -------
        seconds : float
            The number of seconds calculated from the date_string
        """
        date_split = re.split("T|:", date_string)
        return float(int(date_split[1]) * 3600 + int(date_split[2]) * 60)

    @staticmethod
    def distance_check(
                new_point: NDArray | tuple[float, float],
                existing_points: NDArray | list[tuple[float, float]],
                min_dist=10 ** -DISTANCE_GRANULATION
            ) -> bool:
        """
        Check if the new point is within min_dist of any existing points
        """
        if len(existing_points) == 0:
            return True

        if isinstance(new_point, tuple):
            new_point = np.array(new_point, dtype=np.float64)
        if isinstance(existing_points, list):
            existing_points = np.array(existing_points)

        dist = np.sqrt(np.sum((existing_points - new_point)**2, axis=1))
        return bool(np.all(dist >= min_dist))

    def __str__(self) -> str:
        return (
            f"Delivery Environment Info:\n"
            f"\tNumber of targets: {self.n_targets}\n"
            f"\tX axis range: [{self.min_x}, {self.max_x}]\n"
            f"\tY axis range: [{self.min_y}, {self.max_y}]\n"
            f"\tGen prio range: [{self.min_gen_prio}, {self.max_gen_prio}]\n"
            f"\tMax possible distance between points: {self.max_distance}\n"
        )

    def __repr__(self) -> str:
        return (
            f"Delivery Environment Deep Info:\n"
            f"\tTarget Coordinates: {self.all_coords}\n"
            f"\tDistance Matrix: {self.distance_matrix}\n"
            f"\tTime Windows: {self.time_windows}\n"
        )


class DeliveryEnv:
    """Deals with the Environment itself"""

    def __init__(
                self,
                payload: EnvironmentPayload,
            ) -> None:
        """
        Initialize a delivery environment

        Parameters
        ----------
        payload : EnvironmentPayload
            The object containing all the data regarding Environment setup
        """

        self.payload = payload
        self.reset()

        prio_step = 10 ** -PRIORITY_GRANULATION
        dist_step = 10 ** -DISTANCE_GRANULATION
        # Normalized priority from 0 - 1 with a granularly static step
        self.state_space: list[float] = np.round(
            np.arange(
                0,
                1 + prio_step,
                prio_step
            ),
            decimals=PRIORITY_GRANULATION
        ).tolist()

        # Normalized priority from 0 - max prio / min distance
        # with a granularly static step
        self.action_space: list[float] = np.round(
            np.arange(
                0,
                (self.max_gen_prio / dist_step) + dist_step,
                dist_step
            ),
            decimals=DISTANCE_GRANULATION
        ).tolist()

    def get_reward(
                self,
                city: int,
                next_city: int,
                time_left: float | None = None
            ) -> float:

        reward = (
            self.priorities[next_city]
            /
            self.distance_matrix[city, next_city]
        )

        if time_left:
            max_bonus = 100  # TODO max bonus for less time left on obs
            reward += (1 / time_left) if time_left else max_bonus

        return reward

    def reset(self) -> int:
        """
        Reset the environment's schedule memory and returns a random starting
        city (if we have time windows, return the earliest starting city)
        """
        self.schedule: list[int] = []
        
        # If we have time windows, use the earliest start city index
        # The earliest_start is a tuple (time, city_index), so we need to extract the city index
        if self.time_windows is None:
            city = np.random.randint(self.n_targets)
        else:
            # Extract the city index (second element of the tuple)
            city = self.earliest_start[1] if isinstance(self.earliest_start, tuple) else self.earliest_start
            
        self.schedule.append(city)
        return city

    def step(
                self,
                destination: int,
                time_passed: float
            ) -> tuple[int, float, bool]:
        """
        Perform a single step of learning

        Parameters
        ----------
        destination : int
            The index of the city to visit
        time_passed : float
            The amount of time that has passed in the schedule. If given,
            self.time_windows must also exist. Otherwise, nothing occurs.

        Returns
        -------
        int
            The index of the next city
        float
            The reward achieved by visiting this city
        bool
            Whether or not we are now done
        """

        current_city = self.schedule[-1]
        next_city = destination

        reward = self.get_reward(
            current_city,
            next_city,
            (self.time_windows[destination][-1][0] - time_passed)  # TODO
            if self.time_windows is not None
            else None
        )

        self.schedule.append(next_city)
        done = len(self.schedule) == self.n_targets

        return next_city, reward, done

    def format_schedule(self) -> str:
        if len(self.schedule) <= 1:
            raise ValueError("No valid path to format. Must contain 2 items")

        output = ""
        distance = 0
        priority = 0
        time_passed = 0  # Initialize time tracking
        prev = 0
        for next in range(1, len(self.schedule)):
            prev_city = int(self.schedule[prev])
            next_city = int(self.schedule[next])

            distance += (curr_distance := self.distance_matrix[prev_city, next_city])
            priority += self.priorities[next_city]
            
            # Calculate travel time for this segment
            travel_time = self.time_to(curr_distance)
            
            # Add observation time if available
            if self.exptimes is not None and next_city < len(self.exptimes):
                # Use exptime for the next city at the current time
                exptime = self.get_exptime(next_city, time_passed + travel_time)
                travel_time *= exptime  # Adjust travel time by exptime
            
            time_passed += travel_time
            
            distance_str = f"--({curr_distance:.3f})>>"

            if not prev:
                output += f"{prev_city} {distance_str} {next_city}"
            else:
                output += f" {distance_str} {next_city}"

            prev = next

        missed_observations = self.n_targets - len(self.schedule)

        return output + \
            f"\nDistance: {distance}\nPriority: {priority}\nTime Passed: {time_passed:.2f}\n" + \
            f"Missed Observations: {missed_observations}/{self.n_targets}"

    # def validate_schedule(self, schedule: list[int]):
    #     if not self.time_windows:
    #         return True, None

    #     time_passed = self.time_windows[self.earliest_start][0]
    #     current_city = schedule[0]

    #     for next_city in schedule[1:]:
    #         travel_time = self.time_to(
    #             self.distance_matrix[current_city][next_city]
    #         )
    #         future_time = time_passed + travel_time + self.observation_time

    #         if not (
    #             self.time_windows[next_city][0]
    #             <= future_time <=
    #             self.time_windows[next_city][1]
    #         ):
    #             print(f"Violation at city {next_city}:")
    #             print(f"Current time: {future_time}")
    #             print(f"Time window: {self.time_windows[next_city]}")
    #             return False, next_city

    #         current_city = next_city
    #         time_passed = future_time

    #     return True, None

    def show(self):

        # Styling
        plt.style.use("dark_background")

        # Setting up
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Targets")

        # Plotting
        ax.scatter(self.all_x, self.all_y, c="red", s=50)
        plt.xticks([])
        plt.yticks([])

        # Saving/Displaying
        ...
        plt.show()

    def __str__(self) -> str:
        return str(self.payload)

    @staticmethod
    def time_to(distance: float) -> float:
        return distance / TRAVEL_SPEED

    # Pass-Through properties for the Payload object
    @property
    def n_targets(self):
        return self.payload.n_targets

    @property
    def all_x(self):
        return self.payload.all_x

    @property
    def all_y(self):
        return self.payload.all_y

    @property
    def distance_matrix(self):
        return self.payload.distance_matrix

    @property
    def min_gen_prio(self):
        return self.payload.min_gen_prio

    @property
    def max_gen_prio(self):
        return self.payload.max_gen_prio

    @property
    def max_priority(self):
        return self.payload.max_priority

    @property
    def total_priority(self):
        return self.payload.total_priority

    @property
    def priorities(self):
        return self.payload.priorities

    @property
    def time_windows(self):
        return self.payload.time_windows

    @property
    def earliest_start(self):
        return self.payload.earliest_start

    @property
    def exptimes(self):
        """Return the execution time data for each target at different time windows"""
        return self.payload.exptimes

    def observation_time(self, city: int, curr_time: float) -> float:

        if not self.payload.time_windows:
            raise ValueError(
                "Attempting to retrieve an observation time " +
                "but lacking a time window matrix."
            )

        city_delays: list[tuple[float, float]] = self.payload.time_windows[
            city
        ]

        # Too early
        if curr_time < city_delays[0][0]:
            return -1

        # Too late
        if curr_time > city_delays[-1][0]:
            return -1

        # Find current time slice
        prev_start_time: float = city_delays[0][0]
        prev_obs_time: float = city_delays[0][1]
        for (start_time, observation_time) in city_delays:
            if start_time < curr_time:
                prev_start_time = start_time
                prev_obs_time = observation_time
                continue

            if start_time == curr_time:
                return observation_time

            return (
                prev_obs_time +
                (curr_time - prev_start_time)
                *
                (observation_time - prev_obs_time)
                /
                (start_time - prev_start_time)
            )

        return -1

    def get_exptime(self, city: int, curr_time: float) -> float:
        """
        Get the execution time for the given city at the given time.
        
        Parameters
        ----------
        city : int
            The index of the city
        curr_time : float
            The current time
            
        Returns
        -------
        float
            The execution time for the observation at this time. Returns a default value
            if there is no time-dependent data or no matching time window.
        """
        # If no exptime data available, return a default value
        if self.exptimes is None or len(self.exptimes) <= city:
            return 1.0  # Default exptime
            
        city_exptimes = self.exptimes[city]
        
        # Find the closest exptime entry for this time
        best_exptime = 1.0  # Default value
        best_time_diff = float('inf')
        
        for time_window, exptime in city_exptimes:
            # Find the closest time window
            time_diff = abs(time_window - curr_time)
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_exptime = exptime
                
        return best_exptime