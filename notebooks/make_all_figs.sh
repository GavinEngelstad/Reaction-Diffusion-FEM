## runs the files that make the figures in the paper
# config pythonpath to allow src to be accessed
export PYTHONPATH=../:{$PYTHONPATH}
export ROOT=../
export SAVE=True

# trinagulation (Figure 2.1, 2.2)
jupyter nbconvert --execute --to notebook triangulation.ipynb --inplace

# square/circle solutions over time (Figure 3.1)
jupyter nbconvert --execute --to notebook initial_solutions.ipynb --inplace

# parameter choices (Figure 3.2/3.3)
jupyter nbconvert --execute --to notebook parameters.ipynb --inplace

# error (Figure 3.4)
jupyter nbconvert --execute --to notebook error.ipynb --inplace

# alternate domains (Figure 4.1, 4.2
jupyter nbconvert --execute --to notebook alternate_domains.ipynb --inplace
