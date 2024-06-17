# GEOV325: Glaciology

### Simulating the Greenland Ice Sheet Using a Two-Dimensional Ice Flow Model

The two-dimensional ice flow model is written in python. The ice transport equations are numerically
solved in the main model part found in [`2d_model.py`](2d-ice-sheet/2d_model.py), while the SMB is externally
computed and updated in the [`surface_mass_balance.py`](2d-ice-sheet/surface_mass_balance.py) file. The
corresponding parameters for initializing the temperature and precipitation fields are stored
in the `params.py` files in the respective subdirectories.