# Description of modules and their purposes
## PA_CLI
- This module is intended to be a CLI interface for working with the photoacoustic system.
- This module should handle exceptions occured during interaction with underlying modules
## modules.pa_logic
- This module is intended to provide general API for working with the phoroacoustic system, which is independed from interface being used (CLI or GUI)
- Exceptions occured in functions of this module should propagate to the caller functions

## modules.osc_devices
- Нужно выделить в отдельные функции все акты общения с осциллографом (query, write, read, conncetion_check). В случае ошибки возникновения ошибки коммуникации, они должны устанавливать not_found флаг, только эти методы проверяют значения этого флага перед выполнением.
- Методы, входящие в public API должны возвращать статус выполнения операции (True для успешного выполнения, False для ошибки) и значения атрибутов, содержащих результат выполнения метода.
- В случае возникновения ошибки public API методы записывают None в атрибуты, которые должны содержать результаты выполнения методы.
- 