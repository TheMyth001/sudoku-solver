from puzzle_extraction import puzzle_extraction
from digit_recognition import get_numbers_array
from draw_solved import draw
from sudoku import Sudoku
import numpy as np

file = input("Enter file name: ")
sections_array = puzzle_extraction(file)
digits_array = get_numbers_array(sections_array)

puzzle = Sudoku(3, 3, board=list(digits_array))
solution = puzzle.solve()
solution_array = np.array(solution.board)

print(puzzle)
print(solution)

changes = solution_array - digits_array
draw(file, changes)
