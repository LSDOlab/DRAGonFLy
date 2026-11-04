"""
Geometry preparation for the BIG_FISH STEP file (SolidWorks multi-body assembly, mm, axis along z, head at +z).

The assembly (5 segments, joints, head and tail) fuses into a solid with 20 internal cavities and a pair of
coincident, unmerged membrane faces at the head/body junction.  Only the exterior skin matters for the flow, so:
fuse -> merge same-domain faces -> keep the outer shell -> drop duplicated internal planar faces -> sew -> solid
-> scale to metres and centre on the bounding box -> cut from an R = 50 m sphere -> BREP for gmsh.
Needs pythonocc-core (the `dragonfly` conda env).

Usage:  python fish_geometry.py "../../meshes/big_fish_5_segments 1.STEP" out_domain.brep
"""
import sys
from collections import Counter

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing, BRepBuilderAPI_Transform
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.BRepTools import breptools
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.ShapeFix import ShapeFix_Solid
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec

R_FAR = 50.0


def sub(shape, kind):
    out, ex = [], TopExp_Explorer(shape, kind)
    while ex.More():
        out.append(ex.Current()); ex.Next()
    return out


def area(shape):
    g = GProp_GProps(); brepgprop.SurfaceProperties(shape, g); return g.Mass()


def volume(shape):
    g = GProp_GProps(); brepgprop.VolumeProperties(shape, g); return g.Mass()


def bbox(shape):
    b = Bnd_Box(); brepbndlib.Add(shape, b); return b.Get()


def exterior_solid(step_file, min_face_area=0.01, sew_tol=0.01):
    r = STEPControl_Reader(); r.ReadFile(step_file); r.TransferRoots()
    solids = sub(r.OneShape(), TopAbs_SOLID)
    fused = solids[0]
    for s in solids[1:]:
        fused = BRepAlgoAPI_Fuse(fused, s).Shape()
    u = ShapeUpgrade_UnifySameDomain(fused, True, True, True); u.Build()
    outer = max(sub(u.Shape(), TopAbs_SHELL), key=area)
    faces = sub(outer, TopAbs_FACE)

    def key(fc):                     # constant-z planar faces: (z, area)
        if BRepAdaptor_Surface(fc).GetType() != GeomAbs_Plane:
            return None
        bb = bbox(fc)
        return (round(bb[2], 2), round(area(fc))) if abs(bb[5] - bb[2]) < 1e-3 else None

    dup = {k for k, c in Counter(map(key, faces)).items() if k is not None and c >= 2}
    print('internal duplicated planar faces dropped (z[mm], area):', sorted(dup))
    sew = BRepBuilderAPI_Sewing(sew_tol)
    for fc in faces:
        if key(fc) not in dup and area(fc) >= min_face_area:
            sew.Add(fc)
    sew.Perform()
    print(f'sewing: {len(sub(sew.SewedShape(), TopAbs_FACE))} faces, free edges {sew.NbFreeEdges()}, '
          f'multiple edges {sew.NbMultipleEdges()}')
    fix = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(topods.Shell(sub(sew.SewedShape(), TopAbs_SHELL)[0])).Solid())
    fix.Perform()
    solid = fix.Solid()
    assert BRepCheck_Analyzer(solid).IsValid(), 'exterior solid is not valid'
    return solid


def main(step_file, out_brep):
    fish = exterior_solid(step_file)
    t = gp_Trsf(); t.SetScaleFactor(1e-3)                                          # mm -> m
    fish = BRepBuilderAPI_Transform(fish, t, True).Shape()
    x0, y0, z0, x1, y1, z1 = bbox(fish)
    print(f'fish [m]: x {x0:.3f}..{x1:.3f}  y {y0:.3f}..{y1:.3f}  z {z0:.3f}..{z1:.3f}  volume {volume(fish):.5f} m3')
    t = gp_Trsf(); t.SetTranslation(gp_Vec(0, 0, -0.5 * (z0 + z1)))
    fish = BRepBuilderAPI_Transform(fish, t, True).Shape()
    domain = BRepAlgoAPI_Cut(BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), R_FAR).Shape(), fish).Shape()
    assert BRepCheck_Analyzer(domain).IsValid()
    print(f'domain: {len(sub(domain, TopAbs_FACE))} faces, {len(sub(domain, TopAbs_SOLID))} solid(s), volume {volume(domain):.3f} m3')
    breptools.Write(domain, out_brep)


if __name__ == '__main__':
    main(*sys.argv[1:3])
