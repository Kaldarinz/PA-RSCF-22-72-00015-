# Description of modules and their purposes
## PA_CLI
- This module is intended to be a CLI interface for working with the photoacoustic system.
- This module should handle exceptions occured during interaction with underlying modules
## modules.pa_logic
- This module is intended to provide general API for working with the phoroacoustic system, which is independed from interface being used (CLI or GUI)
- Exceptions occured in functions of this module should propagate to the caller functions