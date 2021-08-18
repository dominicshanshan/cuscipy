# https://github.com/scipy/scipy/blob/master/scipy/optimize/minpack2/dcstep.f
import cupy as cp

# params

# double precision vars
zero = cp.array(0, dtype=cp.float64)
p66 = cp.array(0.66, dtype=cp.float64)
two = cp.array(2.0, dtype=cp.float64)
three = cp.array(3.0, dtype=cp.float64)


def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """
    This subroutine computes a safeguarded step for a search procedure and updates an interval that contains a step that
    satisfies a sufficient decrease and a curvature condition

    The parameter `stx` contains the step with the least function value. If brackt is set to `True` then a minimizer has
    been bracketed in an interval with endpoints `stx` and `sty`.
    The parameter `stp` contains the current step.
    The subrouting assumes that if `brackt` is set to `True` then

           min(stx, sty) < stp < max(stx, sty)

    and that the derivative at `stx` is negative in the direction of the step.


    Parameters
    ----------
    stx (double):
    fx (double):
    dx (double):
    sty (double):
    fy (double):
    dy (double):
    stp (double):
    fp (double):
    dp (double):
    brackt (bool):
    stpmin (double):
    stpmax (double):

    Returns
    -------

    """

    sgnd = dp * (dx / cp.abs(dx))

    # First case: A higher function value. The minimum is bracketed.
    # If the cubic step is closer to stx than the quadratic step, the
    # cubic step is taken, otherwise the average of the cubic and
    # quadratic steps is taken.

    if fp > fx:
        theta = three * (fx - fp) / (stp - stx) + dx + dp
        s = cp.array([cp.abs(theta), cp.abs(dx), cp.abs(dp)]).max()
        gamma = s * cp.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))

        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / two) * (stp - stx)

        if cp.abs(stpc - stx) < cp.abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / two

        brackt = True

    # Second case: A lower function value and derivatives of opposite
    # sign. The minimum is bracketed. If the cubic step is farther from
    # stp than the secant step, the cubic step is taken, otherwise the
    # secant step is taken.
    elif sgnd < zero:
        theta = three * (fx - fp) / (stp - stx) + dx + dp
        s = cp.array([cp.abs(theta), cp.abs(dx), cp.abs(dp)]).max()
        gamma = s * cp.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))

        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if cp.abs(stpc - stp) > cp.abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq

        brackt = True

    # Third case: A lower function value, derivatives of the same sign,
    # and the magnitude of the derivative decreases.
    elif cp.abs(dp) < cp.abs(dx):
        # The cubic step is computed only if the cubic tends to infinity
        # in the direction of the step or if the minimum of the cubic
        # is beyond stp. Otherwise the cubic step is defined to be the
        # secant step.
        theta = three * (fx - fp) / (stp - stx) + dx + dp
        s = cp.array([cp.abs(theta), cp.abs(dx), cp.abs(dp)]).max()

        # The case gamma = 0 only arises if the cubic does not tend to infinity in the direction of the step.

        gamma = s * cp.sqrt(cp.array([zero, (theta / s) ** 2 - (dx / s) * (dp / s)]).max())

        if stp > stx:
            gamma = -gamma

        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if (r < zero) and (gamma != zero):
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin

        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            # A minimizer has been bracketed. If the cubic step is
            # closer to stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.

            if cp.abs(stpc - stp) < cp.abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq

            if stp > stx:
                stpf = cp.array([stp + p66 * (sty - stp), stpf]).min()
            else:
                stpf = cp.array([stp + p66 * (sty - stp), stpf]).max()

        else:
            # A minimizer has not been bracketed. If the cubic step is
            # farther from stp than the secant step, the cubic step is
            # taken, otherwise the secant step is taken.
            if cp.abs(stpc - stp) > cp.abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = cp.array([stpmax, stpf]).min()
            stpf = cp.array([stpmin, stpf]).max()

    # Fourth case: A lower function value, derivatives of the same sign,
    # and the magnitude of the derivative does not decrease. If the
    # minimum is not bracketed, the step is either stpmin or stpmax,
    # otherwise the cubic step is taken.
    else:
        if brackt:
            theta = three * (fp - fy) / (sty - stp) + dy + dp
            s = cp.array([cp.abs(theta), cp.abs(dy), cp.abs(dp)]).max()
            gamma = s * cp.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))

            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # update the interval which contains a minimizer
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < zero:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    stp = stpf

    return stx, sty, fx, fy, dx, dy, stp, brackt
