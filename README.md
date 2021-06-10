# CLOSURES

## Installation

Steps to install the `closures` task into `casa`

 1. Clone the git repository into a directory of your choice
 (e.g., $HOME/.casa/NordicTools)

``` shell
cd $HOME/.casa/NordicTools
git clone <repository url>
cd closures
buildmytasks
```
 2. Edit the file `$HOME/.casa/init.py`. Add the line:

``` shell
execfile('$HOME/.casa/NordicTools/closures/mytasks.py')
```

That's it! You should be able to run the new task in CASA! Just doing:

``` shell
tget closures
```

inside `casa` should load the task. To get help, just type `help closures`
