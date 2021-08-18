# https://github.com/scipy/scipy/blob/master/scipy/optimize/minpack2/dcsrch.f

import warnings

import cupy as cp

from .dcstep import dcstep

isave = cp.zeros(2, dtype=cp.int64)
dsave = cp.zeros(13, dtype=cp.float64)

zero = cp.array(0, dtype=cp.float64)
p5 = cp.array(0.5, dtype=cp.float64)
p66 = cp.array(0.66, dtype=cp.float64)
xtrapl = cp.array(1.1, dtype=cp.float64)
xtrapu = cp.array(4.0, dtype=cp.float64)

brackt: bool
stage: cp.int64


def check_args(stp, stpmin, stpmax, g, ftol, gtol, xtol):
    if stp < stpmin:
        raise ValueError('Error: Step < Step Min')
    if stp > stpmax:
        raise ValueError('Error: Step > Step Max')
    if g >= zero:
        raise ValueError('Error: Initial G >= zero')
    if ftol < zero:
        raise ValueError('Error: Ftol < zero')
    if gtol < zero:
        raise ValueError('Error: Gtol < zero')
    if xtol < zero:
        raise ValueError('Error: Xtol < zero')
    if stpmin < zero:
        raise ValueError('Error: Step Min < zero')
    if stpmax < stpmin:
        raise ValueError('Error: Step Max < Step Min')


def check_warnings(brackt, stp, stmin, stmax, xtol, stpmin, stpmax, f, ftest, g, gtest):
    did_warn = False
    if brackt and ((stp <= stmin) or (stp >= stmax)):
        warnings.warn('WARNING: ROUNDING ERRORS PREVENT PROGRESS')
        did_warn = True
    if brackt and ((stmax - stmin) <= (xtol * stmax)):
        warnings.warn('WARNING: XTOL TEST SATISFIED')
        did_warn = True
    if (stp == stpmax) and (f <= ftest) and (g <= gtest):
        warnings.warn('WARNING: STP == STPMAX')
        did_warn = True
    if (stp == stpmin) and ((f > ftest) or (g >= gtest)):
        warnings.warn('WARNING: STP == STPMIN')
        did_warn = True
    return did_warn


def go_to_10(isave, dsave, brackt, stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax, width, width1):
    """
    Instead of Go To 10 in the fortran code, we replace with this function and a return statement afterwards.
    Parameters
    ----------
    isave
    dsave
    brackt
    stage
    ginit
    gtest
    gx
    gy
    finit
    fx
    fy
    stx
    sty
    stmin
    stmax
    width
    width1

    Returns
    -------

    """
    if brackt:
        isave[0] = 1
    else:
        isave[0] = 0

    isave[1] = stage
    dsave[0] = ginit
    dsave[1] = gtest
    dsave[2] = gx
    dsave[3] = gy
    dsave[4] = finit
    dsave[5] = fx
    dsave[6] = fy
    dsave[7] = stx
    dsave[8] = sty
    dsave[9] = stmin
    dsave[10] = stmax
    dsave[11] = width
    dsave[12] = width1
    return isave, dsave


def dcsrch(stp: cp.array, f, g, ftol, gtol, xtol, task, stpmin: cp.array, stpmax: cp.array, isave, dsave):
    """

    Parameters
    ----------
    stp (cp.array):
    f (double):
    g (double):
    ftol (double):
    gtol (double):
    xtol (double):
    task (char): character of length at least 60
    stpmin (cp.array):
    stpmax (cp.array):
    isave (int): integer work array of dim=2
    dsave (double): double precision work array of dim=13

    Returns
    -------

    """

    if task[:5] == b'START':
        check_args(stp, stpmin, stpmax, g, ftol, gtol, xtol)

        # Initialize local variables.

        brackt = False
        stage = 1
        finit = cp.squeeze(f)
        ginit = cp.squeeze(g)
        gtest = ftol * ginit
        width = stpmax - stpmin
        width1 = width / p5

        # The variables stx, fx, gx contain the values of the step,
        # function, and derivative at the best step.
        # The variables sty, fy, gy contain the value of the step,
        # function, and derivative at sty.
        # The variables stp, f, g contain the values of the step,
        # function, and derivative at stp.

        stx = zero
        fx = finit
        gx = ginit
        sty = zero
        fy = finit
        gy = ginit
        stmin = zero
        stmax = stp + xtrapu * stp
        task = b'FG'

        # todo go to 10
        # save local variables
        isave, dsave = go_to_10(isave, dsave, brackt, stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin,
                                stmax, width, width1)
        return stp, f, g, task, isave, dsave

    else:
        # restore local variables

        brackt = True if isave[0] == 1 else False

        stage = isave[1]
        ginit = dsave[0]
        gtest = dsave[1]
        gx = dsave[2]
        gy = dsave[3]
        finit = dsave[4]
        fx = dsave[5]
        fy = dsave[6]
        stx = dsave[7]
        sty = dsave[8]
        stmin = dsave[9]
        stmax = dsave[10]
        width = dsave[11]
        width1 = dsave[12]

    # If psi(stp) <= 0 and f'(stp) >= 0 for some step, then the
    # algorithm enters the second stage.

    ftest = cp.array(finit + stp * gtest)
    if (stage == 1) and (f <= ftest) and (g >= zero):
        stage = 2

    # test for warnings
    did_warn = check_warnings(brackt, stp, stmin, stmax, xtol, stpmin, stpmax, f, ftest, g, gtest)

    # test for convergence
    if (f <= ftest) and (cp.abs(g) <= gtol * (-ginit)):
        task = b'CONVERGENCE'

    # test for termination
    if did_warn or task[:4] == b'CONV':
        # todo go to 10
        # save local variables
        isave, dsave = go_to_10(isave, dsave, brackt, stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin,
                                stmax, width, width1)
        return stp, f, g, task, isave, dsave

    # A modified function is used to predict the step during the
    # first stage if a lower function value has been obtained but
    # the decrease is not sufficient.

    if (stage == 1) and (f <= fx) and (f > ftest):
        # define the modified function and derivative values.

        fm = f - stp * gtest
        fxm = fx - stx * gtest
        fym = fy - sty * gtest
        gm = g - gtest
        gxm = gx - gtest
        gym = gy - gtest

        # call dcstep to update stx, sty, and to compute the new step.

        stx, sty, fx, fy, dx, dy, stp, brackt = dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)

        # reset the function and derivative values for f.

        fx = fxm + stx * gtest
        fy = fym + sty * gtest
        gx = gxm + gtest
        gy = gym + gtest
    else:
        # call dcstep to update stx, sty, and to compute the new step.
        stx, sty, fx, fy, dx, dy, stp, brackt = dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax)

    # decide if a bisection step is needed.

    if brackt:
        if cp.abs(sty-stx) >= p66 * width1:
            stp = stx + p5 * (sty - stx)
        width1 = width
        width = cp.abs(sty - stx)
    # set the minimum and maximum steps allowed for stp.
    if brackt:
        stmin = cp.array([stx, sty]).min()
        stmax = cp.array([stx, sty]).max()
    else:
        stmin = cp.array(stp + xtrapl * (stp - stx))
        stmax = cp.array(stp + xtrapu * (stp - stx))

    # force the step to be within the bounds of stpmax and stpmin

    stp = cp.array([stp, stpmin]).max()
    stp = cp.array([stp, stpmax]).min()

    # If further progress is not possible, let stp be the best point obtained during the search.
    if (brackt and (stp <= stmin or stp >= stmax) or
            (brackt and (stmax-stmin) <= (xtol * stmax) )):
        stp = stx

    # obtain another function and derivative

    task = b'FG'

    isave, dsave = go_to_10(isave, dsave, brackt, stage, ginit, gtest, gx, gy, finit, fx, fy, stx, sty, stmin, stmax,
                            width, width1)

    return stp, f, g, task, isave, dsave
