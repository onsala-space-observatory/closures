# CLOSURES

## Installation

Steps to install the `closures` task into `casa`

 1. Clone the git repository into a directory of your choice
 (e.g., $HOME/.casa/NordicTools)

``` shell
cd $HOME/.casa/NordicTools
git clone <repository url>
cd closures
buildmytasks --module closures closures.xml
```
 2. Inside `casa` add the folder to your `PYTHONPATH`:

``` python
CASA <1>: sys.path.insert(0, <path to closures folder>)
CASA <2>: from closures.gotasks.closures import closures
CASA <3>: inp(closures)

```

That's it! Enjoy!
