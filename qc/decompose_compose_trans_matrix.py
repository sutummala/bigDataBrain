# code for composing and decomposing a transformation matrix
# taken from fslpy, source code for fsl.transform.affine
# Author: Paul McCarthy <pauldmccarthy@gmail.com>

import numpy as np

def concat(*xforms):
    """Combines the given matrices (returns the dot product)."""

    result = xforms[0]

    for i in range(1, len(xforms)):
        result = np.dot(result, xforms[i])

    return result

def axisAnglesToRotMat(xrot, yrot, zrot):
    """Constructs a ``(3, 3)`` rotation matrix from the given angles, which
    must be specified in radians.
    """

    xmat = np.eye(3)
    ymat = np.eye(3)
    zmat = np.eye(3)

    xmat[1, 1] =  np.cos(xrot)
    xmat[1, 2] = -np.sin(xrot)
    xmat[2, 1] =  np.sin(xrot)
    xmat[2, 2] =  np.cos(xrot)

    ymat[0, 0] =  np.cos(yrot)
    ymat[0, 2] =  np.sin(yrot)
    ymat[2, 0] = -np.sin(yrot)
    ymat[2, 2] =  np.cos(yrot)

    zmat[0, 0] =  np.cos(zrot)
    zmat[0, 1] = -np.sin(zrot)
    zmat[1, 0] =  np.sin(zrot)
    zmat[1, 1] =  np.cos(zrot)

    return concat(zmat, ymat, xmat)

def rotMatToAxisAngles(rotmat):
    """Given a ``(3, 3)`` rotation matrix, decomposes the rotations into
    an angle in radians about each axis.
    """

    yrot = np.sqrt(rotmat[0, 0] ** 2 + rotmat[1, 0] ** 2)

    if np.isclose(yrot, 0):
        xrot = np.arctan2(-rotmat[1, 2], rotmat[1, 1])
        yrot = np.arctan2(-rotmat[2, 0], yrot)
        zrot = 0
    else:
        xrot = np.arctan2( rotmat[2, 1], rotmat[2, 2])
        yrot = np.arctan2(-rotmat[2, 0], yrot)
        zrot = np.arctan2( rotmat[1, 0], rotmat[0, 0])

    return [xrot, yrot, zrot]

def compose(scales, offsets, rotations, origin=None):
    """Compose a transformation matrix out of the given scales, offsets
    and axis rotations.

    :arg scales:    Sequence of three scale values.

    :arg offsets:   Sequence of three offset values.

    :arg rotations: Sequence of three rotation values, in radians, or
                    a rotation matrix of shape ``(3, 3)``.

    :arg origin:    Origin of rotation - must be scaled by the ``scales``.
                    If not provided, the rotation origin is ``(0, 0, 0)``.
    """

    preRotate  = np.eye(4)
    postRotate = np.eye(4)

    rotations = np.array(rotations)

    if rotations.shape == (3,):
        rotations = axisAnglesToRotMat(*rotations)

    if origin is not None:
        preRotate[ 0, 3] = -origin[0]
        preRotate[ 1, 3] = -origin[1]
        preRotate[ 2, 3] = -origin[2]
        postRotate[0, 3] =  origin[0]
        postRotate[1, 3] =  origin[1]
        postRotate[2, 3] =  origin[2]

    scale  = np.eye(4, dtype=np.float64)
    offset = np.eye(4, dtype=np.float64)
    rotate = np.eye(4, dtype=np.float64)

    scale[  0,  0] = scales[ 0]
    scale[  1,  1] = scales[ 1]
    scale[  2,  2] = scales[ 2]
    offset[ 0,  3] = offsets[0]
    offset[ 1,  3] = offsets[1]
    offset[ 2,  3] = offsets[2]

    rotate[:3, :3] = rotations

    return concat(offset, postRotate, rotate, preRotate, scale)


def decompose(xform, angles=True):
    """Decomposes the given transformation matrix into separate offsets,
    scales, and rotations, according to the algorithm described in:

    Spencer W. Thomas, Decomposing a matrix into simple transformations, pp
    320-323 in *Graphics Gems II*, James Arvo (editor), Academic Press, 1991,
    ISBN: 0120644819.

    It is assumed that the given transform has no perspective components. Any
    shears in the affine are discarded.

    :arg xform:  A ``(3, 3)`` or ``(4, 4)`` affine transformation matrix.

    :arg angles: If ``True`` (the default), the rotations are returned
                 as axis-angles, in radians. Otherwise, the rotation matrix
                 is returned.

    :returns: The following:

               - A sequence of three scales
               - A sequence of three translations (all ``0`` if ``xform``
                 was a ``(3, 3)`` matrix)
               - A sequence of three rotations, in radians. Or, if
                 ``angles is False``, a rotation matrix.
    """

    # The inline comments in the code below are taken verbatim from
    # the referenced article, [except for notes in square brackets].

    # The next step is to extract the translations. This is trivial;
    # we find t_x = M_{4,1}, t_y = M_{4,2}, and t_z = M_{4,3}. At this
    # point we are left with a 3*3 matrix M' = M_{1..3,1..3}.
    xform = xform.T

    if xform.shape == (4, 4):
        translations = xform[ 3, :3]
        xform        = xform[:3, :3]
    else:
        translations = np.array([0, 0, 0])

    M1 = xform[0]
    M2 = xform[1]
    M3 = xform[2]

    # The process of finding the scaling factors and shear parameters
    # is interleaved. First, find s_x = |M'_1|.
    sx = np.sqrt(np.dot(M1, M1))

    # Then, compute an initial value for the xy shear factor,
    # s_xy = M'_1 * M'_2. (this is too large by the y scaling factor).
    sxy = np.dot(M1, M2)

    # The second row of the matrix is made orthogonal to the first by
    # setting M'_2 = M'_2 - s_xy * M'_1.
    M2 = M2 - sxy * M1

    # Then the y scaling factor, s_y, is the length of the modified
    # second row.
    sy = np.sqrt(np.dot(M2, M2))

    # The second row is normalized, and s_xy is divided by s_y to
    # get its final value.
    M2  = M2  / sy
    sxy = sxy / sy

    # The xz and yz shear factors are computed as in the preceding,
    sxz = np.dot(M1, M3)
    syz = np.dot(M2, M3)

    # the third row is made orthogonal to the first two rows,
    M3 = M3 - sxz * M1 - syz * M2

    # the z scaling factor is computed,
    sz = np.sqrt(np.dot(M3, M3))

    # the third row is normalized, and the xz and yz shear factors are
    # rescaled.
    M3  = M3  / sz
    sxz = sxz / sz
    syz = sxz / sz

    # The resulting matrix now is a pure rotation matrix, except that it
    # might still include a scale factor of -1. If the determinant of the
    # matrix is -1, negate the matrix and all three scaling factors. Call
    # the resulting matrix R.
    #
    # [We do things different here - if the rotation matrix has negative
    #  determinant, the flip is encoded in the x scaling factor.]
    R = np.array([M1, M2, M3])
    if np.linalg.det(R) < 0:
        R[0] = -R[0]
        sx   = -sx

    # Finally, we need to decompose the rotation matrix into a sequence
    # of rotations about the x, y, and z axes. [This is done in the
    # rotMatToAxisAngles function]
    if angles: rotations = rotMatToAxisAngles(R.T)
    else:      rotations = R.T

    return np.array([sx, sy, sz]), translations, rotations

