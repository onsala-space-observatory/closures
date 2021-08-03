import sys
import os

import casatools
from casatasks import simobserve
from casatasks import clearcal
from casatasks import tclean

mydir = os.getenv("HOME") + "/ARC"
sys.path.insert(0, mydir)
from closures.gotasks.closures import closures

vis = "Disc/Disc.alma.out10.noisy.ms"
antennas = ["A00"]

if not os.path.isdir(vis):
    cl = casatools.componentlist()
    ia = casatools.image()
    qa = casatools.quanta()

    # start by simulating an image of a disc
    # Array to use:
    arrayconfig = 'alma.out10.cfg'
    # Center Freq:
    Nu = '50.0GHz'
    # Image size (pixels)
    Npix = 1000
    # Pixel size:
    cell = '0.01arcsec'
    cellN = 0.01
    # Number of frequency channels:
    Nnu = 100
    # Channel width:
    ChW = '0.01GHz'
    imname = 'Disc'
    # type   flux    major       minor       PA  freq   spec.ind.    Direction
    si = ['disk', 1.0, '0.2arcsec', '0.1arcsec', '60.0deg', Nu, 1.0, "J2000 10h00m00.0s -30d00m00.0s"]

    config = '.'.join(arrayconfig.split('.')[:-1])
    print('Generating %s' % imname)
    cl.done()
    cl.addcomponent(dir=si[7], flux=si[1], fluxunit='Jy', freq=si[5], shape=si[0],
                    majoraxis=si[2], minoraxis=si[3], positionangle=si[4],
                    spectrumtype='spectral index', index=si[6])

    ia.fromshape(imname+'.model', [Npix, Npix, 1, Nnu], overwrite=True)
    cs = ia.coordsys()
    cs.setunits(['rad', 'rad', '', 'Hz'])
    cell_rad = qa.convert(qa.quantity(cell), "rad")['value']
    cs.setincrement([-cell_rad, cell_rad], 'direction')
    rdi, ddi = si[7].split()[1:]
    cs.setreferencevalue([qa.convert(rdi, 'rad')['value'],
                          qa.convert(ddi, 'rad')['value']], type="direction")
    cs.setreferencevalue(si[5], 'spectral')
    cs.setincrement(ChW, 'spectral')
    ia.setcoordsys(cs.torecord())
    ia.setbrightnessunit("Jy/pixel")
    ia.modify(cl.torecord(), subtract=False)
    ia.close()

    simobserve(project=imname, skymodel=imname+'.model',
               totaltime='300s', antennalist=arrayconfig)
    Cfile = 'TEST1.CLEAN'
    clearcal('%s/%s.%s.noisy.ms' % (imname, imname, config))

    tclean('%s/%s.%s.noisy.ms' % (imname, imname, config),
           imagename=Cfile, cell=cell,
           imsize=2*Npix, niter=0)

# now load and run closures with all default parameters, except image name
print(vis)
print(antennas)
closures()
