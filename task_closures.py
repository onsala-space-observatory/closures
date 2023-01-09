from auxclos import *
import inspect
from taskinit import *
from itertools import combinations as cb
import numpy as np
import pylab as pl
import os
import sys
import gc
import pickle
import time
tb = gentools(['tb'])[0]
ms = gentools(['ms'])[0]

# Load auxiliary functions:
libloc = inspect.stack()[0][1]
sys.path.append(os.path.dirname(libloc))


def closures(vis='', kind='phase', xaxis='frequency', column='data', field='-1', spw='0',
             scan=-1, timerange='', antennas=[], do_model=False, plot=False,
             histogram=False, outdir='closures.results', xlims=[0, 0]):

    tic = time.time()

    # Set the dictionary to select the data:
    if column not in ['data', 'corrected_data', 'corrected', 'model']:
        print 'Bad string to design data column. Will use \'data\'.'
        column = 'data'
    if column == 'corrected':
        column = 'corrected_data'

    cols = [column, 'flag', 'axis_info', 'time']
    if do_model:
        cols += ['model_data']

    if do_model and column == 'model':
        print '\n It makes no sense to plot model minus model! Aborting!'
        return False

    sdic = {}

    if type(xlims) is not list:
        print 'Bad xlims. it should be a list. Setting it to its default: [0,0]'
        xlims = [0, 0]
    elif len(xlims) != 2:
        print 'Bad xlims. Should have 2 items. Setting it to its default: [0,0]'
        xlims = [0, 0]

    try:
        xlims[0] = int(xlims[0])
        xlims[1] = int(xlims[1])
    except Exception:
        print 'Bad xlims. Should be a list of two integers. Setting it to its default: [0,0]'
        xlims = [0, 0]

    xxlim = [0, 0]
    if xlims[0] <= 0:
        xxlim[0] = {'phase': 180.0, 'amplitude': 10.0}[kind]
    else:
        xxlim[0] = xlims[0]
    if xlims[1] <= 0:
        xxlim[1] = 50
    else:
        xxlim[1] = xlims[1]

    if kind not in ['phase', 'amplitude']:
        print '\n\n   Unkown "kind" parameter. Valid values are "phase" or "amplitude"\n'
        return False

    # Get names of sources and antennas/stations:
    # Get also range of spws:
    tb.open(os.path.join(vis, 'FIELD'))
    sournames = list(tb.getcol('NAME'))
    tb.close()

    tb.open(os.path.join(vis, 'ANTENNA'))
    nants = list(tb.getcol('NAME'))
    statnames = list(tb.getcol('STATION'))
    tb.close()

    tb.open(os.path.join(vis, 'DATA_DESCRIPTION'))
    allspw = list(tb.getcol('SPECTRAL_WINDOW_ID'))
    tb.close()

    tb.open(os.path.join(vis, 'SPECTRAL_WINDOW'))
    allchans = list(tb.getcol('NUM_CHAN'))
    allfreqs = tb.getcol('CHAN_FREQ')
    tb.close()

    # Convert parameters entries into lists:
    sp, chans, success = from_CASA_syntax(spw, [allspw, allchans], 'spw', [0])
    antennas, success2 = from_CASA_syntax(antennas, nants, 'antenna', [])

    if not (success and success2):
        print '\n Check spw and antennas. Aborting!'
        return False


    # Generate lists of all baselines and antennas:
    baselines = []

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plotdir = os.path.join(outdir, 'plots')
    if plot:
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)

    histdir = os.path.join(outdir, 'histograms')
    if histogram:
        if not os.path.exists(histdir):
            os.makedirs(histdir)

    for i in range(len(nants)-1):
        for j in range(i+1, len(nants)):
            baselines.append([nants[i], nants[j]])

    numbas = range(len(baselines))
    numant = range(len(nants))

    antlist = open(os.path.join(outdir, 'antennas.dat'), 'w')

    print >> antlist, '! Index    Antenna name'

    for k, ant in enumerate(nants):
        print >> antlist, '%3i      %10s' % (k, ant)

    antlist.close()

    print '\nOpening dataset.'
    if not ms.open(vis):
        print "Bad measurement set!"
        return False

    # Get either the scan number or the source/timerange pair:
    try:
        scan = int(scan)
    except Exception:
        print 'scan must be an integer. Aborting!'
        return False

    if type(field) is str:
        try:
            field = int(field)
        except Exception:
            pass

    if type(timerange) is not str:
        print '\'timerange\' must be a string. Aborting!'
        return False

    if len(timerange) == 0:
        timerang = []
    else:
        timerang = timerange.split('~')
        if len(timerang) != 2:
            print '\'timerange\' is not formatted as a time range (i.e., like \'t0~t1\'). Aborting!'
            return False

    if (type(antennas) is not list):
        print '\'antennas\' must be a list of strings. Aborting!'
        return False

    if scan >= 0:
        print 'Selecting data based on scan number ', scan
        ms.selectinit(datadescid=sp)
        scrange = map(int, ms.range(['scan_number'])['scan_number'])
        if scan not in scrange:
            print 'Selected scan number (', scan, ') is not in list of scans: ', scrange
            ms.close()
            return False
        else:
            sdic['scan_number'] = scan
            ms.select(sdic)
            mysource = ms.getdata(['field_id'])['field_id'][0]
            print 'This scan is an observation of source: '+sournames[mysource]

    # If the scan number is not set, get the source id...:
    else:
        nofield = False
        if type(field) is int and field > -1:
            if field > len(sournames)-1:
                print '\'field\' parameter is larger than number of fields: ', len(
                    sournames)
                return False
            else:
                sdic['field_id'] = field
                print 'Selected field:', sournames[field]
        elif type(field) is str:
            try:
                sid = sournames.index(field)
                sdic['field_id'] = sid
                print 'Selected field '+field+', with id number', sid
            except Exception:
                print 'Field '+field+' is not found in the data!'
                print 'Valid fields are: ', sournames
                print 'Will try to take the data based on timerange (or just take the first scan).'
                nofield = True
        else:
            print 'No field given (either it is not set, or it is neither an integer nor a string).'
            print 'Will guess source id from the timerange, if possible.'
            nofield = True

        # ...and/or get the timerange:
        notsel = False
        if len(timerang) > 0:
            try:
                t1 = qa.getvalue(qa.convert(qa.quantity(timerang[0]), 's'))
                t2 = qa.getvalue(qa.convert(qa.quantity(timerang[1]), 's'))
                timer = [t1[0], t2[0]]
                sdic['time'] = timer
            except Exception:
                print 'Bad format in timerange string. Will take the source\'s first scan.'
                notsel = True
        else:
            notsel = True

        if notsel:
            print 'Selecting times based on first scan.'
            ms.selectinit(datadescid=sp)
            ms.select(sdic)
            myscan = ms.range(['scan_number'])['scan_number'][0]
            sdic['scan_number'] = int(myscan)
            if nofield:
                ms.selectinit(datadescid=sp)
                ms.select(sdic)
                mysource = ms.getdata(['field_id'])['field_id'][0]
                print 'This scan is an observation of source: ' + \
                    sournames[mysource]


    ms.close()


    # Generate lists of all the antennas, baselines, triplets, ...
    baselines = [list(bl) for bl in cb(nants, 2)]
    triplets = [list(tr) for tr in cb(nants, 3)]
    quads = [list(qd) for qd in cb(nants, 4)]


    # Select only triplets/quads with, at least, the antennas
    # in the list of required antennas (if any)
    if len(antennas) > 0:
        def filtant(x):
            if len(filter(lambda v: v in antennas, x)) == min(len(x), len(antennas)):
                return True
            else:
                return False
        triplets = filter(filtant, triplets)
        quads = filter(filtant, quads)

    Ncombs1 = [len(filter(lambda v: a in v, triplets)) for a in nants]
    Ncombs2 = [len(filter(lambda v: a in v, quads)) for a in nants]

    if plot or histogram:
        fig = pl.figure()
        sub = fig.add_subplot(111)

    out1 = "%2s        %5i      " + "%3i "*{'phase': 3, 'amplitude': 4}[kind]
    output = open(os.path.join(outdir, '%s.closures.dat' % kind), 'w')

    head = '! POL  |  %4s idx.  |  %s    |      closure       |   error' % (
        xaxis[:4], {'phase': 'triplet', 'amplitude': 'quadruplet'}[kind])

    print >> output, head
    outformat = out1 + "   % .11e  % .11e"

    # Get the data (average over time for each baseline):
    if True:

        ms.open(vis)
        ms.selectinit(datadescid=sp)

        print '\nGetting data and averaging.'

        tempsel = ms.select(items=sdic)
        # print sdic
        temp1 = ms.getdata(cols, ifraxis=True, average=False)
        if not tempsel or len(temp1[column]) == 0:
            print 'NO VALID DATA FOR SPW #', sp
            return False

        # Get the polarization labels:
        Pols = list(temp1['axis_info']['corr_axis'])
        Polsymb = ['or', 'ok', 'og', 'ob']

        parhand = []
        if 'XX' in Pols:
            parhand.append(Pols.index('XX'))
        if 'YY' in Pols:
            parhand.append(Pols.index('YY'))
        if 'RR' in Pols:
            parhand.append(Pols.index('RR'))
        if 'LL' in Pols:
            parhand.append(Pols.index('LL'))
        if 'I' in Pols:
            parhand.append(Pols.index('I'))

        # dimensions of the original data.
        NPOL, NCHAN, NBAS, NTIME = np.shape(temp1[column])
        nchan = chans[1]-chans[0]  # channel dimensions to work with.

        print 'Arranging data.'

        CDATA = np.ma.array(temp1[column][:, chans[0]:chans[1], :],
                            mask=temp1['flag'][:, chans[0]:chans[1], :])
        CTIME = temp1['time']
        if do_model:
            MDATA = np.ma.array(temp1['model_data'][:, chans[0]:chans[1], :],
                                mask=temp1['flag'][:, chans[0]:chans[1], :])
        else:
            MDATA = 1.0

        # Output file:
        outxax = {'frequency': 'frequencies.dat',
                  'time': 'juliandates.dat'}[xaxis]
        output2 = open(os.path.join(outdir, outxax), 'w')
        xaxtxt = {'frequency': 'Freq. (GHz)', 'time': 'MJD (days)'}[xaxis]
        yaxtxt = {'phase': 'Phas Clos (deg.)  Error (deg.)', 'amplitude': 'Amp Clos   Error'}[
            kind]

        if xaxis == 'frequency':
            xaxplot = allfreqs[:, sp]/1.e9
            xlabel = 'Frequency (GHz)'
        else:
            xaxplot = CTIME/86400.
            xlabel = 'MJD (days)'

        if kind == 'phase':
            ClosStats = [
                np.zeros((len(parhand), Ncombs1[a], len(xaxplot))) for a in numant]
        else:
            ClosStats = [
                np.zeros((len(parhand), 3*Ncombs2[a], len(xaxplot))) for a in numant]

        Nread = [[0 for a in numant] for b in parhand]

        print >> output2, '! Index    '+xlabel
        for i, xi in enumerate(xaxplot):
            print >> output2, '%5i  %.11e' % (i, xi)

        output2.close()


        # Generate list of antennas and baselines for this scan:
        ANTS = temp1['axis_info']['ifr_axis']['ifr_name']
        BASS = [ANT.split('-') for ANT in ANTS]

        ms.close()

        # Determine baseline ids for the data rows:
        SelBas = []
        for bas in baselines:
            row = filter(lambda x: (bas[0] in x) and (bas[1] in x), BASS)
            if len(row) == 0:
                SelBas.append(-1)  # No data for this baseline
            else:
                SelBas.append(BASS.index(row[0]))

        # Compute all closures:
        if kind == 'phase':
            print '\n Computing closure phases.\n'
            totiter = len(triplets)
        else:
            print '\n Computing closure amplitudes.\n'
            totiter = len(quads)

        nprint = totiter*np.linspace(0, 1, 10)
        pdone = 0

        #  print triplets
        for i in range(totiter):

            done = 10*np.sum(i > nprint)
            if done > pdone:
                sys.stdout.write('\r '+str(done)+'%')
                sys.stdout.flush()
                pdone = done

            # Set of antennas for current closure:
            if kind == 'amplitude':
                q = quads[i]
            elif kind == 'phase':
                q = triplets[i]

            # Compute closure:
            ClosX, NB = ClosComp(CDATA, MDATA, SelBas, baselines, q)
            NA = [nants.index(ant) for ant in q]

            # Update vectors of overall statistics:
            # Otherwise, ClosX[0]=False, and there is no data.
            if type(ClosX[0]) is np.ma.core.MaskedArray:
                #  print len(ClosX), NB[0]
                for nb in range(len(ClosX)):
                    #  for bas in NB[nb]:
                    antxt = '-'.join([nants[k] for k in NA])
                    if plot:
                        sub.cla()
                        sub.set_xlabel(xlabel)
                        pl.title(antxt)
                    if xaxis == "frequency":
                        averclos = np.ma.average(ClosX[nb], axis=2)
                        nelem = np.shape(ClosX[nb])[2]
                        avererr = np.ma.sqrt(np.average(
                            np.power(ClosX[nb]-averclos[:, :, np.newaxis], 2.), axis=2)/nelem)
                    else:
                        averclos = np.ma.average(ClosX[nb], axis=1)
                        nelem = np.shape(ClosX[nb])[1]
                        avererr = np.sqrt(np.average(
                            np.power(ClosX[nb]-averclos[:, np.newaxis, :], 2.), axis=1)/nelem)

                    for pi, pp in enumerate(parhand):
                        x1 = np.shape(averclos[pp, :])[0]
                        for k in NA:
                            # print 'hola 0 ',k,pp,NA,np.shape(ClosStats[k]), Nread[pp][k]
                            ClosStats[k][pi, Nread[pi][k],
                                         :x1] = averclos[pp, :]
                            Nread[pi][k] += 1
                        for j in range(len(averclos[pp, :])):
                            if type(averclos[pp, j]) is np.float64:
                                print >> output, outformat % tuple(
                                    [Pols[pp], j]+NA+[averclos[pp, j], avererr[pp, j]])
                        if plot:
                            sub.plot(xaxplot[:x1], averclos[pp,
                                                            :], Polsymb[pp], label=pol)
                            sub.errorbar(
                                xaxplot[:x1], averclos[pp, :], avererr[pp, :], fmt='k', linestyle='None')

                    if plot:
                        pl.legend(numpoints=1)
                        if kind == 'phase':
                            sub.set_ylabel('Clos. Phase (deg.)')
                            sub.set_ylim((-xxlim[0], xxlim[0]))
                        else:
                            sub.set_ylabel('Clos. Ampl.')
                            sub.set_ylim((1.-xxlim[0]/100., 1.+xxlim[0]/100.))
                        pl.savefig(os.path.join(
                            plotdir, 'PLOT_%s_vs_%s__%s.eps' % (kind[:4], xaxis[:4], antxt)))

    output.close()

    ukind = {'phase': '(deg.)', 'amplitude': '(norm.)'}[kind]
    if histogram:
        print '\n\n Generating histograms'
        histo = np.zeros((x1, xxlim[1]))
        averhisto = np.zeros((len(parhand), len(numant), xxlim[1]))
        if kind == 'phase':
            xran = (-xxlim[0], xxlim[0])
        else:
            xran = (1.-xxlim[0]/100., 1.+xxlim[0]/100.)
        manaspect = np.abs(float(x1)/(xran[1]-xran[0]))
        xlab = {'frequency': 'Freq. channel', 'time': 'Time index'}[xaxis]
        for ant in numant:
            sys.stdout.write(
                '\r Plotting histogram for antenna %i of %i' % (ant, numant[-1]))
            sys.stdout.flush()
            for pi,pp in enumerate(parhand):
                fig.clf()
                sub = fig.add_subplot(111)
                for i in range(x1):
                    histo[i, :] = np.histogram(
                        ClosStats[ant][pi, :Nread[pi][ant], i], bins=xxlim[1], range=xran)[0]
                averhisto[pi, ant, :] = np.average(histo, axis=0)
                ims = sub.imshow(np.transpose(histo), origin='lower', aspect=manaspect,
                                 interpolation='nearest', extent=[0, x1, xran[0], xran[1]])
                sub.set_xlabel(xlab)
                sub.set_ylabel('Closure %s %s' % (kind, ukind))
                pl.title('Antenna %s' % nants[ant])
                colb = pl.colorbar(ims, ax=sub)
                colb.set_label('# of closures')
                pl.savefig(os.path.join(histdir, 'HISTO_%s_vs_%s__%s__%s.eps' % (
                    kind[:4], xaxis[:4], Pols[pp], nants[ant])))

        for pi,pp in enumerate(parhand):
            fig.clf()
            sub = fig.add_subplot(111)
            manaspect = np.abs(float(numant[-1]+1)/(xran[1]-xran[0]))
            ims = sub.imshow(np.transpose(averhisto[pi, :, :]), origin='lower',
                             aspect=manaspect, interpolation='nearest',
                             extent=[0, numant[-1]+0.99, xran[0], xran[1]])
            sub.set_xlabel('Antenna number')
            sub.set_ylabel('Closure %s %s' % (kind, ukind))
            pl.title('Average for %s' % Pols[pp])
            colb = pl.colorbar(ims, ax=sub)
            colb.set_label('# of closures')
            pl.savefig(os.path.join(
                histdir, 'HISTO_%s_ALL_ANTS_%s.eps' % (kind[:4], Pols[pp])))

    stds = np.zeros(len(numant))

    for pp in range(len(parhand)):
        for ant in numant:
            stds[ant] += np.std(ClosStats[ant][pp, :Nread[pp][ant], :])

    print '\n\n---------------------'
    print 'Based on closure statistics, the antennas ordered from BEST to WORSE are:\n'
    for i in np.argsort(stds):
        if stds[i] > 0.0:
            print '%s   -- with std. of %.3e %s' % (nants[i], stds[i], ukind)

    print '---------------------\n'
