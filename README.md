This repository holds the scripts I used to produce the data products and plots included in the paper "Bayesian calibration of quasi-static field distortions in HARMONI".

### Reproducing the results
Although the data products are included, you may want to re-generate them by hand. You will need a GNU/Linux machine with Python3 (I am using 3.11 locally) along with basic dependencies like `numpy`, `matplotlib`, `scipy` and`h5py`.

All tests can be run at once by giving execution permissions to `run_all_tests.sh` and running it as:

```
$ ./run_all_tests.sh
```

The data products are stored in their corresponding test directories.

### Reproducing the plots
The included [Jupyter Lab](https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906?gi=d5c1a2f5971d) notebooks load the data products, display the plots and save them in SVG format. Just open the desired notebook from Jupyter Lab and, under the _Run_ menu, click _Run All Cells_.

### Bugs, errors, issues, etc
These files were originally designed to run in my machine and it was not until much later that I realized I had to release them to the public. Do not hesitate to contact me (either by e-mail or PR) if you run into trouble reproducing the results.