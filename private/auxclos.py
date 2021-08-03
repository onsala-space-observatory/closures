import numpy as np

def from_CASA_syntax(list0, totlist, whatami, default):
    """In case the user applies 'CASA-like' syntax,
    this function returns the correct formatted
    lists (either of antennas, channels, spws,...).
    It also checks if the list contents are right."""

    found = False
    list1 = default
    lstype = type(totlist[0])

    if whatami == 'spw':

        if ',' in list0:
            print('Only one spw is allowed at a time!')
            return 0, 0, False

        if ':' in list0:
            colind = list0.index(':')
            try:
                myspw = int(list0[:colind])
            except:
                print('Bad spw syntax!')
                return 0, 0, False
            chran = list0[colind+1:].split('~')
            try:
                ch0 = int(chran[0])
                if len(chran) == 2:
                    ch1 = int(chran[1])+1
                else:
                    ch1 = ch0+1
            except:
                print('Bad spw syntax!')
                return 0, 0, False
            if ch0 < totlist[1][myspw] and ch1 < totlist[1][myspw] and ch0 < ch1:
                return myspw, [ch0, ch1], True
            else:
                print('Bad channel range! Will take all the spw!')
                return myspw, [0, totlist[1][myspw]-1], True

        else:
            try:
                myspw = int(list0)
            except:
                print('Bad spw syntax!')
                return 0, 0, False
            return myspw, [0, totlist[1][myspw]], True

    elif whatami == 'antenna':

        try:
            # check if all elements exist
            indices = [totlist.index(elem) for elem in list0]
            list1 = [totlist[index] for index in indices]  # Assign elements.
            return list1, True
        except:
            print('Bad antenna list! Check that all antennas are in the MS!')
            return 0, False

def ClosComp(CDATA, MDATA, SelBas, baselines, q):
    """Compute the (residual) phase or amplitude closures
       for a particular triplet/quadruplet"""

    if len(q) == 3:  # triplet. Hence, phase closure.
        phase = True
        basels = [[q[0], q[1]], [q[1], q[2]], [q[0], q[2]]]
    else:  # Quadruplet. Hence, amplitude closure.
        phase = False
        basels = [[q[0], q[1]], [q[2], q[3]], [q[0], q[2]],
                  [q[1], q[3]], [q[0], q[3]], [q[1], q[2]]]

    b = [baselines.index(bas) for bas in basels]

    irows = [SelBas[bi] for bi in b]  # Row numbers for the required baselines.

    # One or more baselines are missing.
    if len(list(filter(lambda x: x < 0, irows))) > 0:
        return [[False], []]   # Returns a null result.

    else:  # All baselines have data.

        Xvis = [CDATA[:, :, irow] for irow in irows]
        # There aren't model data. Use a point source.
        if type(MDATA) is float:
            XvisMod = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:  # There are model data. Use them.
            XvisMod = [MDATA[:, :, irow] for irow in irows]

        if phase:  # closure phases (1 closure per triplet)

            BisPecX = Xvis[0]*Xvis[1]*np.conjugate(Xvis[2])
            BisPecX *= np.conjugate(XvisMod[0] *
                                    XvisMod[1]*np.conjugate(XvisMod[2]))
            return [[180./np.pi*np.arctan2(np.imag(BisPecX), np.real(BisPecX))], [b]]

        else:  # closure amplitudes (3 closures per quadruplet)

            AbsInt = map(np.abs, Xvis)
            ClosX1 = AbsInt[0]*AbsInt[1]/(AbsInt[4]*AbsInt[5])
            ClosX2 = AbsInt[2]*AbsInt[3]/(AbsInt[4]*AbsInt[5])
            ClosX3 = ClosX1/ClosX2

            AbsInt = map(np.abs, XvisMod)
            ClosModX1 = AbsInt[0]*AbsInt[1]/(AbsInt[3]*AbsInt[4])
            ClosModX2 = AbsInt[2]*AbsInt[3]/(AbsInt[4]*AbsInt[5])
            ClosModX3 = ClosModX1/ClosModX2

            # Compute the closure-amplitude residuals
            # (define closures such that the residuals are >=0.0, on average):
            ClosRat1 = ClosModX1/ClosX1
            ClosRat2 = ClosModX2/ClosX2
            ClosRat3 = ClosModX3/ClosX3

            if np.sum(ClosRat1) > np.sum(1./ClosRat1):
                ClosX1 = ClosRat1  # - 1.0
            else:
                ClosX1 = 1./ClosRat1  # - 1.0
            if np.sum(ClosRat2) > np.sum(1./ClosRat2):
                ClosX2 = ClosRat2  # - 1.0
            else:
                ClosX2 = 1./ClosRat2  # - 1.0
            if np.sum(ClosRat3) > np.sum(1./ClosRat3):
                ClosX3 = ClosRat3  # - 1.0
            else:
                ClosX3 = 1./ClosRat3  # - 1.0

            b1 = [b[0], b[1], b[3], b[4]]
            b2 = [b[2], b[3], b[4], b[5]]
            b3 = [b[0], b[1], b[2], b[3]]

            return [[ClosX1, ClosX2, ClosX3], [b1, b2, b3]]
