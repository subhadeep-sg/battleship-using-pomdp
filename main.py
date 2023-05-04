from random import randint
from time import sleep
import numpy as np
import warnings
from statistics import mean

warnings.filterwarnings('ignore')

# Settings to display less or more information
verbose = True
add_delay = False
final_board_and_probabilities = True


# Ship Class
class Ship:
    def __init__(self, size, orientation, location):
        self.size = size
        self.coordinates = []

        if orientation == 'horizontal' or orientation == 'vertical':
            self.orientation = orientation
        else:
            raise ValueError("Value must be 'horizontal' or 'vertical'.")

        if orientation == 'horizontal':
            if location['row'] in range(row_size):
                self.coordinates = []
                for index in range(size):
                    if location['col'] + index in range(col_size):
                        self.coordinates.append({'row': location['row'], 'col': location['col'] + index})
                    else:
                        raise IndexError("Column is out of range.")
            else:
                raise IndexError("Row is out of range.")
        elif orientation == 'vertical':
            if location['col'] in range(col_size):
                self.coordinates = []
                for index in range(size):
                    if location['row'] + index in range(row_size):
                        self.coordinates.append({'row': location['row'] + index, 'col': location['col']})
                    else:
                        raise IndexError("Row is out of range.")
            else:
                raise IndexError("Column is out of range.")

        if self.filled():
            print_board(board)
            print(" ".join(str(coords) for coords in self.coordinates))
            raise IndexError("A ship already occupies that space.")
        else:
            self.fillBoard()

    def filled(self):
        for coords in self.coordinates:
            if board[coords['row']][coords['col']] == 1:
                return True
        return False

    def fillBoard(self):
        for coords in self.coordinates:
            board[coords['row']][coords['col']] = 1

    def contains(self, location):
        for coords in self.coordinates:
            if coords == location:
                return True
        return False

    def destroyed(self):
        for coords in self.coordinates:
            if board_display[coords['row']][coords['col']] == 'O':
                return False
            elif board_display[coords['row']][coords['col']] == '*':
                raise RuntimeError("Board display inaccurate")
        return True

    def updateDisplay(self):
        for coords in self.coordinates:
            board_display[coords['row']][coords['col']] = '1'


def possible_configs(size, orientation, check_coords=None, focus=None):
    """
    Function to find relevant ways of placing a ship around a given coordinate
    and all possible ways to place the ship everywhere else on the board
    Then returns a list of both all configurations and relevant configurations
    :param size: int
        Size of ship
    :param orientation: str
        Orientation of the ship, either horizontal or vertical
    :param check_coords: dict of str : int
        Coordinates of a particular cell to check configurations on
    :param focus: bool
        Whether the cells near the hit ship cells should be prioritized
    :return: list of dict, list of dict
        locations : Gives a list of all possible coordinates where the ship can be places
        coord_locations : Gives a list of all relevant coordinates where ship can be placed
    """
    if check_coords is None:
        x, y = -2, -2
    else:
        x = check_coords['row']
        y = check_coords['col']

    locations = []
    coord_locations = []

    if orientation != 'horizontal' and orientation != 'vertical':
        raise ValueError("Orientation must have a value of either 'horizontal' or 'vertical'.")

    if focus and orientation == 'horizontal':
        if size <= col_size:
            for r in range(row_size):
                for c in range(col_size - size + 1):
                    # if ('*' not in board_display[r][c:c + size] or '1' not in board_display[r][c:c + size]) \
                    #         and r == x:
                    if ('X' in board_display[r][c:c + size] and 'O' in board_display[r][c:c + size]) and r == x:
                        locations.append({'row': r, 'col': c})
                        if c <= y <= (c + size):
                            coord_locations.append({'row': r, 'col': c})
        return locations, coord_locations

    if focus and orientation == 'vertical':
        if size <= row_size:
            for c in range(col_size):
                for r in range(row_size - size + 1):
                    # if ('*' not in [board_display[k][c] for k in range(r, r + size)] or \
                    #         '1' not in [board_display[i][c] for i in range(r, r + size)]) and c == y:
                    if 'X' in [board_display[k][c] for k in range(r, r + size)] \
                            and 'O' in [board_display[k][c] for k in range(r, r + size)] and c == y:
                        locations.append({'row': r, 'col': c})
                        if r <= x <= (r + size):
                            coord_locations.append({'row': r, 'col': c})
        return locations, coord_locations

    if orientation == 'horizontal':
        if size <= col_size:
            for r in range(row_size):
                for c in range(col_size - size + 1):
                    if 'X' in board_display[r][c:c + size] and 'O' in board_display[r][c:c + size]:
                        locations.append({'row': r, 'col': c})
                        if r == x and c <= y <= (c + size):
                            coord_locations.append({'row': r, 'col': c})
        return locations, coord_locations

    if orientation == 'vertical':
        if size <= row_size:
            for c in range(col_size):
                for r in range(row_size - size + 1):
                    if 'X' in [board_display[k][c] for k in range(r, r + size)] \
                            and 'O' in [board_display[k][c] for k in range(r, r + size)]:
                        locations.append({'row': r, 'col': c})
                        if r <= x <= (r + size) and c == y:
                            coord_locations.append({'row': r, 'col': c})
        return locations, coord_locations

    if not locations:
        return 'None'


def probability_count(m, n, verb='None', focus=False):
    """
    Function to give the probability of placing all ships of all sizes
    based on relevant configurations and all possible configurations
    :param m: int
        Row coordinate
    :param n: int
        Column coordinate
    :param verb: str
        Verbose: whether function should explicitly state the configurations
    :param focus: bool
        Whether the cells near the hit ship cells should be prioritized
    :return: float
        Ratio of relevant configurations by all possible configurations
    """
    relevant_count = 0
    total_count = 0

    check_coords = {'row': m, 'col': n}

    if board_display[m][n] == 'X' or board_display[m][n] == '*':
        return 0

    for size in range(min_ship_size, max_ship_size + 1):
        if size not in ignore_list:
            locations, coord_locations = possible_configs(size, 'vertical', check_coords, focus)
            locations2, coord_locations2 = possible_configs(size, 'horizontal', check_coords, focus)

            if locations != 'None':
                total_count += len(locations)

            if locations2 != 'None':
                total_count += len(locations2)

            if coord_locations != 'None':
                relevant_count += len(coord_locations)

            if coord_locations2 != 'None':
                relevant_count += len(coord_locations2)

    if verb == 'show counts':
        # print('There were {} relevant configurations'.format(relevant_count))
        # print('There were {} possible configurations'.format(total_count))
        print('There were {} ways to place ships'.format(relevant_count))

    if total_count == 0:
        return 0

    # return relevant_count / total_count
    return relevant_count / total_count


class RandomPlayer:
    """
    Class for the random player
    Attributes
    ----------
    hit_count : int
        keeps a counter of hit parts of a ship
    hit_coords : list of int
        list of coordinates where a ship was hit
    """
    def __init__(self):
        self.hit_count = 0
        self.hit_coords = []

    def play_moves(self):
        """
        function to output the move that the random player wants to play.
        Randomly initializes row and column index and checks if given cell is yet to be attacked
        :return:
        dict of {str: int}
        str: row or column
        int: index value
        """
        i, j = 0, 0
        while board_display[i][j] != 'O':
            i = randint(0, row_size - 1)
            j = randint(0, col_size - 1)
        return {'row': i, 'col': j}


class OneStepLookahead(Ship):
    """
    Class for OneStepLookAhead algorithm
    Attributes
    ----------
    hit_count : int
        keeps a counter of hit parts of a ship
    hit_coords : list of int
        list of coordinates where a ship was hit
    probability_representations: float
        matrix keeping track of probability values calculated on cells.

    Methods
    -------
    printPR():
        Prints the probability representation matrix
    playStrategy():
        Selects the coordinates that it wants to attack
        Initially picks a random coordinate and computes the probability of ship
        existing on that cell.
        Then compares this probability with those of other cells and finds the cell
        with the highest probability.
        :return:
        dict of str:int
        str: row or column
        int: index value
    """
    def __init__(self):
        self.hit_coords = []
        self.hit_count = 0

        self.probability_representations = np.round(board, 3)

    def printPR(self):
        print("\n  \t" + "\t \t ".join(str(x) for x in range(1, col_size + 1)))
        for r in range(row_size):
            print(str(r + 1) + "\t " + "\t ".join(str(c) for c in self.probability_representations[r]))
        print()

        #

    def playStrategy(self):
        i, j = 0, 0
        while board_display[i][j] != 'O':
            i = randint(0, row_size - 1)
            j = randint(0, col_size - 1)
        new_coords = {'row': i, 'col': j}
        big = probability_count(i, j)
        self.probability_representations[i][j] = np.round(big, 4)

        if self.hit_count > 0:
            big = 0.0
            for coordinates in self.hit_coords:
                i = coordinates['row']
                j = coordinates['col']
                for m in range(row_size):
                    if board_display[m][j] == 'O' and big < probability_count(m, j):
                        big = probability_count(m, j, True)
                        self.probability_representations[i][j] = np.round(big, 4)
                        new_coords = {'row': m, 'col': j}

                for m in range(col_size):
                    if board_display[i][m] == 'O' and big < probability_count(i, m):
                        big = probability_count(i, m, True)
                        self.probability_representations[i][j] = np.round(big, 4)
                        new_coords = {'row': i, 'col': m}

                return new_coords

        else:
            for i in range(row_size):
                for j in range(col_size):
                    if board_display[i][j] == 'O' and probability_count(i, j) > big:
                        new_coords = {'row': i, 'col': j}
                        big = probability_count(i, j)
                        self.probability_representations[i][j] = np.round(big, 4)
        if verbose:
            _ = probability_count(new_coords['row'], new_coords['col'], 'show counts')
        return new_coords


class POMdp(Ship):
    """
    Class for the Rollout algorithm (Two step look ahead)
    Attributes
    ----------
    ships_occupy : int
        cells with ships
    water_space : int
        cells with water
    miss_prob : float
        probability of missing and hitting the water on a given coordinate and board state
    hit_prob : float
        probability of hitting a ship on a given coordinate and board state
    r_hit : float
        reward for hitting part of a ship
    r_miss : int
        reward for missing and hitting the water
    hit_count : int
        keeps a counter of hit parts of a ship
    hit_coords : list of int
        list of coordinates where a ship was hit
    probability_representations: float
        matrix keeping track of probability values calculated on cells.

    Methods
    -------
    printPR():
        Prints the probability representation matrix

    cond_probability(next_coords, current_coords, observation)
        Function to return the probability of hitting or missing a ship
        based on a given observation
        :arg
        next_coords : dict of str: int
            Gives the coordinates of the next cell
        current_coords : dict of str:int
            Gives coordinates of current cell
        observation : str
            Specifies the observation. Whether ship was hit or not.
            Additionally if hit given previous observation
            or miss given previous observation
        :return:
        prob : float
            Probability value calculated

    value_function(i, j):
        Function that returns the expected value based on the probability
        and reward of hitting or missing a ship
        :arg
        i : int
            Row coordinate of current cell
        j : int
            Column coordinate of current cell
        :return:
         int
         Sum of reward on current state and reward on next state (Two step lookahead)

    play_coords():
        Generates the coordinates of the ship the algorithm wants to attack
        Computes value function for all cells and picks the one giving the highest
        value.
        :return:
        move : dict of str : int
    """
    def __init__(self):
        self.ships_occupy = sum(range(min_ship_size, max_ship_size + 1))
        self.water_space = (row_size * col_size) - self.ships_occupy
        self.miss_prob = self.water_space / (self.water_space + self.ships_occupy)
        self.hit_prob = self.ships_occupy / (self.water_space + self.ships_occupy)
        self.r_hit = 0.5
        self.r_miss = -1
        self.hit_count = 0
        self.hit_coords = []
        self.probability_representations = np.round(board, 3)
        self.gamma = 1

    def printPR(self):
        print("\n  \t" + "\t \t ".join(str(x) for x in range(1, col_size + 1)))
        for r in range(row_size):
            print(str(r + 1) + "\t " + "\t ".join(str(c) for c in self.probability_representations[r]))
        print()

    def cond_probability(self, next_coords, current_coords, observation='None'):
        m = current_coords['row']
        n = current_coords['col']

        if board_display[m][n] == 'X' or board_display[m][n] == '*':
            return 0.0

        if observation == 'miss':
            prob = 1 - probability_count(m, n, 'None', True)

        elif observation == 'hit':
            prob = probability_count(m, n, 'None', True)

        elif observation == 'hitgivenhit' or observation == 'missgivenhit':
            # Temporarily assuming that the current coordinate is in fact a hit
            self.hit_coords.append(current_coords)
            self.hit_count += 1
            board_display[m][n] = 'X'

            hit_prob = probability_count(next_coords['row'], next_coords['col'], 'None', True)
            miss_prob = 1.0 - hit_prob

            # Reverting the changes made to the board and hit coordinates list
            self.hit_coords.remove(current_coords)
            self.hit_count -= 1
            board_display[m][n] = 'O'

            if observation == 'hitgivenhit':
                return hit_prob
            else:
                return miss_prob

        elif observation == 'hitgivenmiss' or observation == 'missgivenmiss':
            # Temporarily assuming that the current coordinate is in fact a miss
            board_display[m][n] = '*'

            hit_prob = probability_count(next_coords['row'], next_coords['col'])
            miss_prob = 1.0 - probability_count(next_coords['row'], next_coords['col'])

            # Reverting the changes made to the board
            board_display[m][n] = 'O'

            if observation == 'hitgivenmiss':
                return hit_prob
            else:
                return miss_prob

        # if verb:
        #     print('Probability of {} is {}'.format(observation, prob))
        return prob

    def value_function(self, i, j):
        term1 = 0.0
        term2 = 0.0
        # Current state coordinates are i and j
        current_coords = {'row': i, 'col': j}

        # Defining probability of missing and hitting to avoid redundant computation
        p_miss = self.cond_probability(None, current_coords, 'miss', False, True)
        p_hit = self.cond_probability(None, current_coords, 'hit', False, True)

        rs = (p_hit * self.r_hit) + (p_miss * self.r_miss)

        for ni in range(row_size):
            for nj in range(col_size):
                if board_display[ni][nj] == 'O':
                    next_coords = {'row': ni, 'col': nj}
                    term1 += (self.cond_probability(next_coords, current_coords, 'hitgivenhit', False, True) * p_hit) \
                        + (self.cond_probability(next_coords, current_coords, 'hitgivenmiss', False,
                                                      True) * p_miss)

                    term2 += self.cond_probability(next_coords, current_coords, 'missgivenhit', False, True) * p_hit \
                        + self.cond_probability(next_coords, current_coords, 'missgivenmiss', False, True) * p_miss

        return rs + self.gamma * (self.r_hit * np.round(term1, 4) + self.r_miss * np.round(term2, 4))

    def play_coords(self):
        i, j = 0, 0
        while board_display[i][j] != 'O':
            i = randint(0, row_size - 1)
            j = randint(0, col_size - 1)
        move = {'row': i, 'col': j}
        big = self.value_function(i, j)
        self.probability_representations[i][j] = np.round(big, 2)

        for i in range(row_size):
            for j in range(col_size):
                if board_display[i][j] == 'O' and self.value_function(i, j) > big:
                    move = {'row': i, 'col': j}
                    big = self.value_function(i, j)
                    self.probability_representations[i][j] = np.round(big, 2)

        # print('Value function = {} on coordinates ({},{})'.format(big, move['row'], move['col']))
        return move


# Settings Variables
row_size = 6  # number of rows
col_size = 6  # number of columns
num_ships = 4
max_ship_size = 5
min_ship_size = 2
num_turns = 80

# Create lists
ship_list = []

# List of ship sizes that are already destroyed and don't need to be accounted for
ignore_list = []

board = [[0.0] * col_size for x in range(row_size)]

board_display = [["O"] * col_size for x in range(row_size)]


# Functions
def print_board(board_array):
    print("\n  " + " ".join(str(x) for x in range(1, col_size + 1)))
    for r in range(row_size):
        print(str(r + 1) + " " + " ".join(str(c) for c in board_array[r]))
    print()


def search_locations(size, orientation):
    """
    Finds all the points on the game board where the ship of give
    size and orientation can be placed
    :param size: int
        Size of ship
    :param orientation: str
        Orientation of ship, either horizontal or vertical
    :return: locations :list of dict {str : int}
        List of coordinates
    """
    locations = []

    if orientation != 'horizontal' and orientation != 'vertical':
        raise ValueError("Orientation must have a value of either 'horizontal' or 'vertical'.")

    if orientation == 'horizontal':
        if size <= col_size:
            for r in range(row_size):
                for c in range(col_size - size + 1):
                    if 1 not in board[r][c:c + size]:
                        locations.append({'row': r, 'col': c})

    elif orientation == 'vertical':
        if size <= row_size:
            for c in range(col_size):
                for r in range(row_size - size + 1):
                    if 1 not in [board[i][c] for i in range(r, r + size)]:
                        locations.append({'row': r, 'col': c})

    if not locations:
        return 'None'
    else:
        return locations


def random_location(size):
    """
    Functions to get one random location from all possible coordinates
    where the ship of given size can be placed
    :param size: int
        Size of ship
    :return:
     dict{str : dict of str: int,
        str : int,
        str: str}
        Location with coordinates
        Size
        Orientation
    """
    # size = randint(min_ship_size, max_ship_size)

    # Fixing ships size and number to make it easier for probabilities

    orientation = 'horizontal' if randint(0, 1) == 0 else 'vertical'

    locations = search_locations(size, orientation)
    if locations == 'None':
        return 'None'
    else:
        return {'location': locations[randint(0, len(locations) - 1)],
                'size': size,
                'orientation': orientation}


# Get row input from user
# def get_row():
#     while True:
#         try:
#             guess = int(input("Row Guess: "))
#             if guess in range(1, row_size + 1):
#                 return guess - 1
#             else:
#                 print("\nOops, that's not even in the ocean.")
#         except ValueError:
#             print("\nPlease enter a number")


# Get column input from user
# def get_col():
#     while True:
#         try:
#             guess = int(input("Column Guess: "))
#             if guess in range(1, col_size + 1):
#                 return guess - 1
#             else:
#                 print("\nOops, that's not even in the ocean.")
#         except ValueError:
#             print("\nPlease enter a number")

# Setting up the list of results on experiments
turnlist_random = []
turnlist_OSLA = []
turnlist_POMDP = []

# Setting number of games to compare over.
num_of_games = 1

for game in range(num_of_games):
    ship_list = []

    # List of ship sizes that are already destroyed and don't need to be accounted for
    ignore_list = []

    board = [[0.0] * col_size for x in range(row_size)]

    board_display = [["O"] * col_size for x in range(row_size)]
    # Create the ships (initialized at random locations)
    # Both experiments run on the same game board to ensure that algorithms are being compared fairly
    temp = 0
    while temp < num_ships:
        for size_ship in range(min_ship_size, max_ship_size + 1):
            ship_info = random_location(size_ship)
            if ship_info == 'None':
                continue
            else:
                ship_list.append(Ship(ship_info['size'], ship_info['orientation'], ship_info['location']))
                temp += 1
    del temp

    ship_list_copy = ship_list.copy()
    ship_list_second_copy = ship_list.copy()

    # Play Game
    # Starting position of game board
    if verbose:
        print_board(board_display)

    # Initializing objects to play the algorithms
    play1 = POMdp()
    play2 = OneStepLookahead()
    play3 = RandomPlayer()

    # First loop of num_turns for the Random Player
    for turn in range(num_turns):
        if verbose:
            print("Turn:", turn + 1, "of", num_turns)
            print("Ships left:", len(ship_list_second_copy))
            # Adding delay to visually see the progress of the game
            if add_delay:
                sleep(1.5)
            print()

        guess_coords = play3.play_moves()
        ship_hit = False

        for ship in ship_list_second_copy:
            if ship.contains(guess_coords):
                if verbose:
                    print("Hit!")
                ship_hit = True
                board_display[guess_coords['row']][guess_coords['col']] = 'X'
                if ship.destroyed():
                    if verbose:
                        print('Ship of size {}, has been destroyed!'.format(ship.size))
                    ship_list_second_copy.remove(ship)
                    # Marks the destroyed ship with '1' to distinguish between half and fully sunk ships
                    ship.updateDisplay()
                break

        if not ship_hit:
            board_display[guess_coords['row']][guess_coords['col']] = '*'
            if verbose:
                print("You missed!")
        # Display of board state at the end of current turn
        if verbose:
            print_board(board_display)
        # If ship list is empty, then all ships have been sunk
        if not ship_list_second_copy:
            break

        # End Game
    if ship_list_second_copy:
        print("You lose!")
        print("You took {} turns and still lost. ".format(turn + 1))

    else:
        print("All the ships are sunk! You took {} turns to win. [Random] ".format(turn))
        turnlist_random.append(turn)

    # List of ship sizes that are already destroyed and don't need to be accounted for
    ignore_list = []
    board = [[0.0] * col_size for x in range(row_size)]
    board_display = [["O"] * col_size for x in range(row_size)]
    # Loop for specified number of turns in the game
    # Second loop of num_turns is used for the One Step Lookahead Algorithm
    for turn in range(num_turns):
        if verbose:
            print("Turn:", turn + 1, "of", num_turns)
            print("Ships left:", len(ship_list))
            # Adding delay to visually see the progress of the game
            if add_delay:
                sleep(1.5)
            print()

        guess_coords = play2.playStrategy()

        while True:
            # Input from user. Disabled for current purposes
            # guess_coords['row'] = get_row()
            # guess_coords['col'] = get_col()

            # Checking for redundant input. Serves as a check for errors from algorithm in giving input.
            if board_display[guess_coords['row']][guess_coords['col']] == 'X' or \
                    board_display[guess_coords['row']][guess_coords['col']] == '*':
                print("\nYou guessed that one already.")
            else:
                break

        # After coordinates have been input by user or algorithm, checking with each ship to see whether hit
        ship_hit = False

        for ship in ship_list:
            if ship.contains(guess_coords):
                # Adding coordinates of hit ships parts to a list to allow heuristic algorithm to check nearby cells
                play2.hit_coords.append(guess_coords)
                play2.hit_count += 1
                if verbose:
                    print("Hit!")
                ship_hit = True
                board_display[guess_coords['row']][guess_coords['col']] = 'X'
                if ship.destroyed():
                    if verbose:
                        print('Ship of size {}, has been destroyed!'.format(ship.size))
                    ship_list.remove(ship)
                    # Adding an ignore list so that the possible configurations
                    # don't take into account the size of sunk ship
                    ignore_list.append(ship.size)
                    for coords in ship.coordinates:
                        play2.hit_coords.remove(coords)
                    # Marks the destroyed ship with '1' to distinguish between half and fully sunk ships
                    ship.updateDisplay()
                    # Delete the size of sunk ship from flag counting hit parts of ships.
                    play2.hit_count -= ship.size
                break

        if not ship_hit:
            board_display[guess_coords['row']][guess_coords['col']] = '*'
            if verbose:
                print("You missed!")
        # Display of board state at the end of current turn
        if verbose:
            print_board(board_display)
        # If ship list is empty, then all ships have been sunk
        if not ship_list:
            break

    # End Game
    if ship_list:
        print("You lose!")
        print("You took {} turns and still lost. ".format(turn + 1))

    else:
        print("All the ships are sunk! You took {} turns to win. [OSLA] ".format(turn))
        turnlist_OSLA.append(turn)

    # Re-initializing list and boards for second loop using POMDP
    ignore_list = []
    board = [[0.0] * col_size for x in range(row_size)]
    board_display = [["O"] * col_size for x in range(row_size)]

    # Third loop of num_turns is for the POMDP algorithm
    for turn in range(num_turns):
        if verbose:
            print("Turn:", turn + 1, "of", num_turns)
            print("Ships left:", len(ship_list_copy))
            # Adding delay to visually see the progress of the game
            if add_delay:
                sleep(1.5)
            print()

        guess_coords = play1.play_coords()

        while True:
            # Input from user. Disabled for current purposes
            # guess_coords['row'] = get_row()
            # guess_coords['col'] = get_col()

            # Checking for redundant input. Serves as a check for errors from algorithm in giving input.
            if board_display[guess_coords['row']][guess_coords['col']] == 'X' or \
                    board_display[guess_coords['row']][guess_coords['col']] == '*':
                print("\nYou guessed that one already.")
            else:
                break

        # After coordinates have been input by user or algorithm, checking with each ship to see whether hit
        ship_hit = False

        for ship in ship_list_copy:
            if ship.contains(guess_coords):
                # Adding coordinates of hit ships parts to a list to allow heuristic algorithm to check nearby cells
                play1.hit_coords.append(guess_coords)
                play1.hit_count += 1
                if verbose:
                    print("Hit!")
                ship_hit = True
                board_display[guess_coords['row']][guess_coords['col']] = 'X'
                if ship.destroyed():
                    if verbose:
                        print('Ship of size {}, has been destroyed!'.format(ship.size))
                    ship_list_copy.remove(ship)
                    # Adding an ignore list so that the possible configurations
                    # don't take into account the size of sunk ship
                    ignore_list.append(ship.size)
                    for coords in ship.coordinates:
                        play1.hit_coords.remove(coords)
                    # Marks the destroyed ship with '1' to distinguish between half and fully sunk ships
                    ship.updateDisplay()
                    # Delete the size of sunk ship from flag counting hit parts of ships.
                    play1.hit_count -= ship.size
                break

        if not ship_hit:
            board_display[guess_coords['row']][guess_coords['col']] = '*'
            if verbose:
                print("You missed!")
        # Display of board state at the end of current turn
        if verbose:
            print_board(board_display)
        # If ship list is empty, then all ships have been sunk
        if not ship_list_copy:
            break

    # End Game
    if ship_list_copy:
        print("You lose!")
        print("You took {} turns and still lost. ".format(turn + 1))

    else:
        print("All the ships are sunk! You took {} turns to win. [POMDP] ".format(turn))
        turnlist_POMDP.append(turn)

print('Average of {} games: {} turns to win using Random Player'.format(num_of_games, mean(turnlist_random)))
print('Average of {} games: {} turns to win using One Step Look Ahead'.format(num_of_games, mean(turnlist_OSLA)))
print('Average of {} games: {} turns to win using POMDP'.format(num_of_games, mean(turnlist_POMDP)))
# Final board display and the associated probabilities of each cell taken into consideration by the algorithm
if final_board_and_probabilities:
    print_board(board_display)
    print('')
    print('Value functions calculated for POMDP')
    play1.printPR()
    print('Probability beliefs calculated for OSLA')
    play2.printPR()
