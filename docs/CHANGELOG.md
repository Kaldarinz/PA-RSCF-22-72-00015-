# Change Log

## Unreleased

### Added

* To be added...

### Changed

#### Data storage and representation

* 

### Depricated

* To be depricated...

### Fixed

* To be fixed...

## 1.1.1 - [2023-10-08]

### Added

#### General

* Changelog created

#### Data storage and representation

* Measured data can now be annotated with notes. System will ask to add notes during `save data` procedure.
  The notes are displayed in `view data` as data description.

#### Hardware operation

* Added stage identification. During the identification stage is vibrating near its current position and stage's controoler is blinking.
  It allows one to verify that the system properly assigned mechanical stages to their corresponding axes (i.e. that 'Y' is movement of US sensor UP and DOWN, not LEFT to RIGHT).
  If the assignment is wrong, it is possible to reassign the stages on the fly.
  This feature can be found in `Init and status` -> `Assign stage axes`.

### Changed

#### General

* `logs` folder now keeps only 10 last *.log file.
  When a new log file is created the oldest one is removed.

#### Data storage and representation

* PA data is now stored relative to the begining of laser pulse.
  It means that data from US sensor is measured between `x_var_start` and `x_var_stop`, which have values relative to the begining of laser pulse.

### Fixed

#### Data storage and representation

* `Export to txt` now properly works for storing `raw data`, `filtered data` and `frequency data` of 0D and 1D types (point measurements and spectral measurements).
* `View data` now properly works with 0D and 1D data (point measurements and spectral measurements).
