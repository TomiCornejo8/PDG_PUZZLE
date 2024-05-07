import numpy as np
from modules import mechanics as M

def scenarioFiller(n, m, expansion_factor):
    """
    Improved generation of an n x m dungeon filled with zeros and ones ensuring all zeros are cross-connected.
    This function also includes the validation step using flood fill algorithm.

    Parameters:
    n (int): Number of rows.
    m (int): Number of columns.
    expansion_factor (float): Probability of a '1' expanding towards the center.

    Returns:
    np.ndarray: The filled dungeon if it is valid, otherwise it tries again.
    """

    def is_valid_matrix(dungeon):
        """
        Check if the dungeon is valid based on the flood fill algorithm.
        """
        n, m = dungeon.shape
        visited = np.zeros_like(dungeon, dtype=bool)

        def flood_fill(r, c):
            """
            Perform flood fill to mark connected zeros using an iterative approach with a stack.
            """
            # Stack to hold the cells to process
            stack = [(r, c)]

            # Loop until there are no more cells to process
            while stack:
                current_r, current_c = stack.pop()

                # Boundary and base case checks
                if current_r < 0 or current_r >= n or current_c < 0 or current_c >= m:
                    continue
                if visited[current_r, current_c] or dungeon[current_r, current_c] != 0:
                    continue

                # Mark as visited
                visited[current_r, current_c] = True

                # Add adjacent cells to stack
                stack.append((current_r + 1, current_c))  # down
                stack.append((current_r - 1, current_c))  # up
                stack.append((current_r, current_c + 1))  # right
                stack.append((current_r, current_c - 1))  # left

        # Start flood fill from the first zero found
        for i in range(n):
            for j in range(m):
                if dungeon[i, j] == M.EMPTY:
                    flood_fill(i, j)
                    break
            if dungeon[i, j] == M.EMPTY:
                break

        # If there are unvisited zeros, the dungeon is not valid
        for i in range(n):
            for j in range(m):
                if dungeon[i, j] == M.EMPTY and not visited[i, j]:
                    return False

        return True

    # Keep generating matrices until a valid one is found
    while True:

        # Create the initial dungeon with 0s
        dungeon = np.zeros((n, m))

        # Fill the contour with 1s
        dungeon[0, :] = M.WALL
        dungeon[:, 0] = M.WALL
        dungeon[-1, :] = M.WALL
        dungeon[:, -1] = M.WALL

        # Calculate the maximum expansion length (25% of the minimum dimension)
        max_expansion_length = min(n, m) // 4

        # Expansion process
        for _ in range(max_expansion_length+1):
            indices_to_expand = np.where(dungeon == M.WALL)
            for row, col in zip(*indices_to_expand):
                if np.random.rand() < expansion_factor:
                    if row > 0 and dungeon[row - 1, col] == M.EMPTY:
                        dungeon[row - 1, col] = 1
                    if row < n - 1 and dungeon[row + 1, col] == M.EMPTY:
                        dungeon[row + 1, col] = 1
                    if col > 0 and dungeon[row, col - 1] == M.EMPTY:
                        dungeon[row, col - 1] = 1
                    if col < m - 1 and dungeon[row, col + 1] == M.EMPTY:
                        dungeon[row, col + 1] = 1

        if is_valid_matrix(dungeon):
            return dungeon