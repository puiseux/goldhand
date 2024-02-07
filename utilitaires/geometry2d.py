#!/usr/local/bin/python
#-*-coding: utf-8 -*-

"""
Created on 11 mai 2012

@author: puiseux
"""
# import math
import pylab
import numpy as np
from pyvista import Plotter, PolyData, lines_from_points
from debog import debug, stack#, mexit
from shapely import speedups
from utils import dist, computeCordeAndNBA, absCurv, diff, pointsDoublesConsecutifs
from shapely.geometry.linestring import LineString
speedups.enable()
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from math import sqrt, atan2
from scipy.interpolate.interpolate import interp1d
from scipy.interpolate import CubicSpline
from numpy.linalg import norm
from scipy.interpolate import (InterpolatedUnivariateSpline, UnivariateSpline)
from numpy import (sin, cos, pi, where,vstack, ones, ndarray, asarray, matrix,
                   copy, nan, prod, inf, zeros, linspace, hstack)
from numpy import asarray as array, insert

def longueur(X:np.ndarray):
    """
    Longueur du polyligne X.
    """
    return sum(norm(X[1:]-X[:-1], axis=1))


# from matplotlib import pyplot as plt
def hardScale(points, echelle, centre=None):#, translate=False):
    """
    (2D) homothétie de centre 'centre': X = C + ech*(X-C) = (1-ech)*C + ech*X
    points est modifié in situ.
    """
#     if centre is None :
#         centre=[0,0]
#     centre=asarray(centre).reshape((1,2))
    points *= asarray([echelle[0],echelle[1]])
    if centre is not None :
        centre = asarray(centre)
#         debug(centre)
        centre.shape = (1,2)
        points += [1.0-echelle[0], 1.0-echelle[1]]*centre
    return points


def rayCross(polygon, point):
    """ détermine si point est à l'intérieur ou a l'extérieur du polygône
    par la technique de "ray-crossing"
        The essence of the ray-crossing method is as follows.
        Think of standing inside a field with a fence representing the polygon.
        Then walk north. If you have to jump the fence you know you are now
        outside the poly. If you have to cross again you know you are now
        inside again; i.e., if you were inside the field to start with, the total
        number of fence jumps you would make will be odd, whereas if you were
        ouside the jumps will be even.

        The code below is from Wm. Randolph Franklin <wrf@ecse.rpi.edu>
        (see URL below) with some minor modifications for speed.  It returns
        1 for strictly interior points, 0 for strictly exterior, and 0 or 1
        for points on the boundary.  The boundary behavior is complex but
        determined; in particular, for a partition of a region into polygons,
        each point is "in" exactly one polygon.
        (See p.243 of [O'Rourke (C)] for a discussion of boundary behavior.)

        int pnpoly(int npol, float *xp, float *yp, float x, float y)
        {
          int i, j, c = 0;
          for (i = 0, j = npol-1; i < npol; j = i++) {
            if ((((yp[i]<=y) && (y<yp[j])) ||
                 ((yp[j]<=y) && (y<yp[i]))) &&
                (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))

              c = !c;
          }
          return c;
        }
    """
    point, polygon
    print('todo...')

def isInside(p, Q):
    """
    Paramètres:
    ==========
        - p = (x,y) = np.ndarray((1,2), float) un point en 2d
        - Q est un polygone = np.ndarray((n,2), float) supposé fermé i.e. le premier point == le dernier point,
    Retourne :
    ========
        - vrai si p est à l'intérieur du polygone Q et
        - faux s'il est à l'extérieur ou frontiere
    Fonctionnement :
    ==============
    p est intérieur, si et seulement si les produits vectoriels AB^Ap ont tous le même signe.
    (on note A=Q[i], B=Q[i+1], 0<= i < n )
    """
#    debug(p,Q)
#     (x, y) = p
#    debug(x,y)
    w0 = vect2d(Q[1]-Q[0], p-Q[0])
    for (A, B) in zip(Q[:-1], Q[1:]) :
        #[A,B] parcourt les segments de Q
        w =  vect2d(B-A, p-A)
#        debug(w0, w)
        if w*w0 <= 0 : return False

    return True

def isInside0(p, Q, frontiere=False):
    """
    Paramètres:
    ==========
        - p = (x,y) = np.ndarray((1,2), float) un point en 2d
        - Q est un quadrilatère = np.ndarray((5,2), float) avec deux cotés parallèles à l'axe des y
            Q est supposé fermé i.e. le premier point == le dernier point,
            on n'utilise que les points 0 à 3 notés A, B, C, D
        - ATTENTION à l'ordre des points : cf dessin ci-dessous.
          A x¨ ¨ - - _ _
            |            ¨ ¨ -x B
            |                 |
            |      . p        |
            |                 |
            |         _ _ --  x C
          D x - - ¨ ¨
        - TODO : frontiere=False pour indiquer qu'un point frontière est considéré comme extérieur.
    Retourne :
    ========
        - vrai si p est à l'intérieur du quadrilatere Q = ABCDA et
        - faux s'il est à l'extérieur.
    Fonctionnement :
    ==============
    pour être intérieur, p doit vérifier les conditions suivantes :
        - xA < x <xB
        - les deux produits vectoriels AB^Ap et CD^Cp  ont même signe.
    """
#    debug(p,Q)
    frontiere
    (x, _) = p
#    debug(x,y)
    A, B, C, D = Q[0], Q[1], Q[2], Q[3]
#    debug(A,B,C,D)
    if x <= A[0] or x >= B[0] :
#        print '****x'
        return False

    wa = vect2d(B-A, p-A)
    wf = vect2d(D-C, p-C)
#    print wa, wf
    if wa*wf > 0 : return True# meme signe  => point interieur
    return False

def isOn(P, segment, eps=1.0e-10):#, debug=False):
    """
    Retourne True si P est sur le segment, à eps près
    Paramètres :
    ----------
    - segment : [A,B] ou A et B de type np.ndarray((1,2),dtype=float)
    - P : de type np.ndarray((1,2),dtype=float)
    - eps : de type float >= 0.0
    Subject 1.02: How do I find the distance from a point to a line?
    ===============================================================
    Projection orthogonale d'un point sur un segment

    Let the point be C (Cx,Cy) and the line be AB (Ax,Ay) to (Bx,By).
    Let P be the point of perpendicular projection of C on AB.  The parameter
    r, which indicates P's position along AB, is computed by the dot product
    of AC and AB divided by the square of the length of AB:

    (1)     AC dot AB
        r = ---------
            ||AB||^2

    r has the following meaning:

        r=0      P = A
        r=1      P = B
        r<0      P is on the backward extension of AB
        r>1      P is on the forward extension of AB
        0<r<1    P is interior to AB

    The length of a line segment AB is computed by:
        L = sqrt( (Bx-Ax)^2 + (By-Ay)^2 )
    and the dot product of two vectors , U dot V is computed:
        D = (Ux * Vx) + (Uy * Vy)
    So (1) expands to:
            (Cx-Ax)(Bx-Ax) + (Cy-Ay)(By-Ay)
        r = -------------------------------
                          L^2

    The point P can then be found:

        Px = Ax + r(Bx-Ax)
        Py = Ay + r(By-Ay)
    """
    A, B = segment[0], segment[1]
    x, y = P - A
    u, v = B - A
    det_, dot_ = abs(u*y - v*x), (x*u + y*v)/(u*u + v*v)
    val = det_ < eps and -eps <= dot_ <= 1.0 + eps
#    if debug : return val, det_, dot_
#    else :
    return val

def isInsideOrFrontier(p, Q, eps=1.0e-6):
    """
    Paramètres:
    ==========
        - p = (x,y) = np.ndarray((1,2), float) un point en 2d
        - Q est un polygone = np.ndarray((n,2), float) supposé fermé i.e. le premier point == le dernier point,
        - eps : precision
    Retourne :
    ========
        - vrai si p est à l'intérieur du polygone Q ou sur la frontiere (bande de largeur eps) et
        - faux s'il est à l'extérieur.
    """
    P = Polygon(Q)
    p = Point(p)
    #debug(p=Point(p), lr=lr)
#     debug(distance=P.distance(p) - p.distance(P))
    return P.distance(Point(p))<eps

# def absCurv(points, normalise=False):
#     """Abscisse curviligne des points de points, considéré comme polyligne"""
# #     debug(points)
# #     stack()
#     l = len(points)
#     if l==0 : return []
#     elif l==1 : return [0]
#     T = zeros(l)
#     for k in range(l-1) :
#         T[k+1] = T[k] + dist(points[k],points[k+1])
#     if normalise and T[-1] != 0.0 :
#         T /= T[-1]
#         #sinon T=[0,0,...] On peut très bien entrer ici avec 3 points = 0
#     return T

def intersectionSegments(seg1, seg2):
    """http://www.exaflop.org/docs/cgafaq/cga1.html :
    Subject 1.03: How do I find intersections of 2 2D line segments?

This problem can be extremely easy or extremely difficult depends on your applications.
If all you want is the intersection point, the following should work:

Let A,B,C,D be 2-space position vectors.  Then the directed line segments AB & CD are given by:

        AB=A+r(B-A), r in [0,1]
        CD=C+s(D-C), s in [0,1]

If AB & CD intersect, then

        A+r(B-A)=C+s(D-C), or

        Ax+r(Bx-Ax)=Cx+s(Dx-Cx)
        Ay+r(By-Ay)=Cy+s(Dy-Cy)  for some r,s in [0,1]
Solving the above for r and s yields

            (Ay-Cy)(Dx-Cx)-(Ax-Cx)(Dy-Cy)
        r = -----------------------------  (eqn 1)
            (Bx-Ax)(Dy-Cy)-(By-Ay)(Dx-Cx)
            (Ay-Cy)(Bx-Ax)-(Ax-Cx)(By-Ay)
        s = -----------------------------  (eqn 2)
            (Bx-Ax)(Dy-Cy)-(By-Ay)(Dx-Cx)
Let P be the position vector of the intersection point, then

        P=A+r(B-A) or

        Px=Ax+r(Bx-Ax)
        Py=Ay+r(By-Ay)
By examining the values of r & s, you can also determine some other limiting conditions:

        If 0<=r<=1 & 0<=s<=1, intersection exists
            r<0 or r>1 or s<0 or s>1 line segments do not intersect
If the denominator in eqn 1 is zero, AB & CD are parallel

If the numerator in eqn 1 is also zero, AB & CD are coincident

If the intersection point of the 2 lines are needed (lines in this context mean infinite lines) regardless whether the two line segments intersect, then

If r>1, P is located on extension of AB

If r<0, P is located on extension of BA

If s>1, P is located on extension of CD

If s<0, P is located on extension of DC

Also note that the denominators of eqn 1 & 2 are identical.
    """
#     print seg1, seg2
    A,B=seg1# np.ndarray((2,))
    u=B-A
    C,D=seg2
    v=D-C

    w = C-A #=AC
    denominateur=u[0]*v[1]-u[1]*v[0]
    if denominateur==0.0 :
        return np.nan,np.nan,np.asarray((np.nan,np.nan))
    else  :
        r=(w[0]*v[1]-w[1]*v[0])/denominateur
        s=(w[0]*u[1]-w[1]*u[0])/denominateur
        return r,s,A+r*u

def segmentPlusProche(points,P):
    """
    Paramètres:
    ----------
        - points : np.ndarray((n,2)) est un polyligne dans lequel on cherche le segment
        - P : np.ndarray((1,2)) est un point quelconque
    Fonctionnement:
    --------------
        On recherche le segment S[i]=(points[i], points[i+1]) le plus proche de P au sens suivant :
        pour tous les segments (A,B)=(points[i], points[i+1]), P' désigne la projection orthogonale
        de P sur la droite AB. Si P' est INTERIEURE au segment [A,B], le segment est candidat
        Parmi tous les segments candidats, on retient celui qui réalise la plus courte distance PP'.
    Retourne :
    --------
        - (i, P') i=numéro du segment et P'=le projeté de P.
        - (None, None) s'il n'y a pas de candidat.

Voir sur http://www.exaflop.org/docs/cgafaq/cga1.html#Subject%201.02:%20How%20do%20I%20find%20the%20distance%20from%20a%20point%20to%20a%20line?

Subject 1.02: How do I find the distance from a point to a line?

Let the point be C (Cx,Cy) and the line be AB (Ax,Ay) to (Bx,By).    The length of the line segment AB is L:

    L= sqrt( (Bx-Ax)^2 + (By-Ay)^2 ) .

Let P be the point of perpendicular projection of C onto AB. Let r be a parameter to indicate P's location along the line containing AB, with the following meaning:

      r=0      P = A
      r=1      P = B
      r<0      P is on the backward extension of AB
      r>1      P is on the forward extension of AB
      0<r<1    P is interior to AB

Compute r with this:

        (Ay-Cy)(Ay-By)-(Ax-Cx)(Bx-Ax)
    r = -----------------------------
                    L^2

The point P can then be found:

    Px = Ax + r(Bx-Ax)
    Py = Ay + r(By-Ay)

And the distance from A to P = r*L.

Use another parameter s to indicate the location along PC, with the following meaning:

       s<0      C is left of AB
       s>0      C is right of AB
       s=0      C is on AB

Compute s as follows:

        (Ay-Cy)(Bx-Ax)-(Ax-Cx)(By-Ay)
    s = -----------------------------
                    L^2

Then the distance from C to P = s*L.
    """

#         if isinstance(P,(QPointF, QPoint,)):
#             P = P.x(), P.y()# Point argument
    points = np.asarray(points)
    u = -(points-P)[:-1]
    v = points[1:]-points[:-1]
#         debug(v=v)
#         distances = [dist(point,(X,Y)) for point in points]
#         debug(points-P)
#         debug(distances=distances)
#         debug(P, len(points), distances)
#         distances = np.linalg.norm(u, axis=1)
    longueurs = np.linalg.norm(v, axis=1)
#         debug(distances_P_points=distances)
#         debug(longueurs_segments=longueurs)
    ps = [np.dot(ui, vi) for (ui, vi) in zip(u,v)]
    r = ps/(longueurs*longueurs)
#         debug(r_entre_0_et_1=r)
#         psn = [np.dot(ui, vi)/np.dot(vi,vi) for (ui, vi) in zip(u,v)]
#         debug(psn=psn)
#         distances = r*longueurs
#         debug(distances_P_Segment=distances)
    candidats, = np.where(np.logical_and(0.0<=r,r<=1.0))
#         debug(len(points), len(r), len(v))
#         if len(candidats)>0 :
#             projetes = points[candidats]+r[candidats]*v[candidats]
    distances = []
    projetes = []
    for candidat in candidats :
#                 debug(candidat=candidat)
#                 debug(r_candidat=r[candidat])
#                 debug(p_candidat=points[candidat])
#                 debug(v_candidat=v[candidat])
        H = points[candidat]+r[candidat]*v[candidat]
        projetes.append(H)
        distances.append(dist(P,H))

    if len(candidats) == 0 :
        pp, distances = pointLePlusProche(points, P, return_distances_array=True)
        #parcequ'il faut bien retourner quelque chose
#             if pp == 0 : pp=1#segment p[0], p[1] <= bug
        if pp == len(points)-1 : pp = len(points)-2#segment p[-2], p[-1]
        return pp, None
    else :
        winner = np.argmin(distances)
#             debug(winner=winner)
#             debug(candidats_winner=candidats[winner])
        i = candidats[winner]
        projete = projetes[winner]
#         debug(i=i, projete=projete)
        return i, projete

def pointLePlusProche(points,P,return_distances_array=False):
    """
    Parametres:
    ----------

    :param points: np.ndarray, tableau de n points 2d de shape (n,2)
    :param P: un point 2d de type QPointF, QPoint ou tuple ou liste (x,y)
    :param return_distances_array: bool, retourne ou non le tableau des distances.

    :return: (i, dist) avec

    - i : l'indice dans 'points' du point qui réalise le min
        - dist : la distance de P à points si return_distances_array est False
        - dist : le tableau des distances de P à points si return_distances_array est True
    """
#     if isinstance(P,(QPointF,QPoint,)):
#         X ,Y=P.x(),P.y()# Point argument
#     else :
    X,Y=P[0],P[1]
    distances=[dist(point,(X,Y)) for point in points]
    index=np.argmin(distances)
    if return_distances_array :
        return index,distances
    else :
        return index,distances[index]

def longueur2d(polyline):
    P = np.asarray(polyline)
    return sum([dist(p1,p2) for (p1, p2) in zip(P[1:], P[:-1])])

def splineInterpolation(points, methode='c cubic', tension=5, degre=3):
    """
    Une paire de spline cubiques paramétriques qui interpole ou ajuste le polygone points
    sx(t), sy(t) sont deux splines.
    Voir la doc Scipy.interpolate...
    - si methode dans {'x cubic', 'ius','periodic','interp1d',} c'est de l'interpolation
    - si methode est {'us','univariatespline'} c'est de l'ajustement, le poids est 1 pour tous les points
    Retourne:
    --------
    T = les valeurs du paramètre t, abscisse curviligne NORMALISEE, entre 0 et 1.
    sx, sy = les deux splines
    """
    if methode in ('periodic', 'p cubic', ) :
        if all(points[0] == points[-1]) : pass
        else : #On rajoute le premier point a la fin
            points = vstack((points, points[0]))
#             points.resize((1+len(points),2)) #moins cher que vstack mais marche pas !!
#             points[-1] = points[0]
    if tension == 0.0 :
        eps = 0.0
    else :
        eps = 10.0**(-tension)

    N = len(points)
    T = absCurv(points, normalise=True)
    if len(points)<2 : return T, None, None
#     T /= T[-1]
    X = points[:,0]
    Y = points[:,1]
    try : methode = methode.lower()
    except AttributeError : pass
#     debug(None, methode=methode, tension=tension, degre=degre)
    if methode in ('ius','interpolatedunivariatespline') :
        try :
            sx = InterpolatedUnivariateSpline(T, X, k=degre)#s=la précision de l'ajustement s=0 <=> interpolation
            sy = InterpolatedUnivariateSpline(T, Y, k=degre)
        except Exception as msg:
            stack(str(msg))
            nums = pointsDoublesConsecutifs(T)
            print('nums_points_doubles =',nums)
#             debug(None,u'Impossible de calculer la testspline (pas assez de points ?, degré trop élévé ?)')
            sx = sy = None
    elif methode in ('us','univariatespline') :
        try :
            weights = ones(N)
            W = 1000.0
            # en supposant que tous les termes erreur di^2=wi*(xi-f(ti))^2 sont egaux
            # le choix de s suivant implique
            # abs(xi-f(ti))<eps et
            # abs(x1-f(t1))<eps/(N*W) et abs(xN-f(tN))<eps/(N*W)
#             eps = 10.0**(-tension)
            weights[0] = weights[-1] = W
            weights /= sum(weights)
            s = eps/(N*W)
            sx = UnivariateSpline(T, X, w=weights, k=degre, s=s)#s=la précision de l'ajustement s=0 <=> interpolation
            sy = UnivariateSpline(T, Y, w=weights, k=degre, s=s)
        except Exception as msg:
            debug(None)
            print((str(msg)))
#             debug(None,u'Impossible de calculer la testspline (pas assez de points ?, degré trop élévé ?)')
            sx = sy = None
    elif methode in ('interp1d',) :
        try :
            sx = interp1d(T, X, kind=degre)
            sy = interp1d(T, Y, kind=degre)
        except ValueError as msg:
            debug(None)
            print((str(msg)))
            sx = sy = None
#     elif methode in ('periodic',) :
#         try :
#             sx = PeriodicSpline(T, X, k=degre, s=eps)
#             sy = PeriodicSpline(T, Y, k=degre, s=eps)
#         except ValueError as msg:
#             debug(None)
#             print unicode(msg)
#             sx = sy = None
    elif 'cubic' in methode :#or isinstance(methode, (tuple, list, np.ndarray)):
        if methode == 'p cubic' : bc_type='periodic'
        elif methode == 'c cubic' : bc_type='clamped'
        elif methode == 'n cubic' : bc_type='natural'
        else : bc_type = 'not-a-knot'

        try :
#             debug(None, T)
            sx = CubicSpline(T, X, bc_type=bc_type)
            sy = CubicSpline(T, Y, bc_type=bc_type)
        except ValueError as msg:
            print((str(msg)))
            sx = sy = None

    elif isinstance(methode, (tuple, list, ndarray)):
        bc_type = methode
        try :
#             debug(None, T)
            sx = CubicSpline(T, X, bc_type=bc_type)
            sy = CubicSpline(T, Y, bc_type=bc_type)
        except ValueError as msg:
            print((str(msg)))
            sx = sy = None
    return T, sx, sy

def rotated(points, alfa, centre=(0.,0.)):
    """
    alfa en radians
     Retourne une COPIE de points rotationnée(!).
    points est supposé stocké par ligne (shape=(n,2)), chaque point est de shape (1,2),
    il les faudrait en colonne (shape=(2,1)) pour faire le produit matriciel.
    Donc on transpose tout et on ecrit Xi' = C' + (Xi'-C')*A' au lieu de
    Xi = C + A*(Xi-C), pour i= 0, 1,...
    """
    Ct = asarray(centre).reshape((1,2))
    cosa, sina = cos(alfa), sin(alfa)
    At = matrix([[cosa,-sina], [sina,cosa]]).transpose()
    Xt = points - Ct
    Xt = Xt*At + Ct
    return asarray(Xt)

def symetrieAxe(points,axe,centre=0.0):
    """
    symetrie
    - d'axe 'axe' = 0 (vertical) ou 1 (horizontal)
    - de centre 'centre', réel
    X = 2*centre - X pour axe vertical
    Y = 2*centre - Y pour axe horizontal
    points est modifié in situ.
    """
#     debug(points.shape, axe, centre)
    points.shape=(-1,2)
    points[:,axe]*=-1
    if centre !=0.0 : points[:,axe]+=2*centre
    return points


def aire2d(points,absolute=True):
    """Aire algébrique ou absolue délimitée par les points de points.
    = 0.5*det(AP(i), AP(i+1)) 1<=i<n avec points = {P[i], 0<=i<=n}, et A=P[0]"""
#    if isinstance(points, QPolygonF) :
#        points = pointsFromPolygon(points)
    if len(points)<=2 :
        return 0.0
    points=copy(points)#sinon effet de bord difficile à identifier !!!
    S=0.0
    A=points[0]
    points-=A
    if absolute :
        for b,c in zip(points[1:-2],points[2:-1]) :
            d=abs(b[0]*c[1]-b[1]*c[0])
            S+=d
    else :
        for b,c in zip(points[1:-2],points[2:-1]) :
            d=b[0]*c[1]-b[1]*c[0]
            S+=d
    return 0.5*S

def centreGravite(points, surface=False):
    """
    Centre de gravité du polygone délimité par les points de 'points'
    Le polygone EST une plaque plane (de densité surfacique constante),
    Le polygone N'EST PAS un ensemble de cotés de masse linéique constante,
    Le polygone N'EST PAS un ensemble de masses constantes disposées aux sommets.
    La méthode utilisée est
    -de fixer un point O quelconque,
    -de remplacer chaque triangle t(i)=OP(i)P(i+1) par une masse ponctuelle proportionnelle à sa surface S(t(i))
        située aux centre de gravité du triangle.
    -puis de faire le barycentre de ces masses affectées de la surface du triangle.

    Que le polygone soit fermé ou non ne change pas le résultat.
    On obtient :
    S*OG = som(S(ti)*OG(ti), pout ti parcourant les triangles ti=OP(i)P(i+1) 0<=i<n
    S est l'aire algebrique
    S(ti) est l'aire algébrique du triangle ti , i.e. 0.5*det(OP(i), OP(i+1))
    """
    if len(points) == 0 :
        if surface : return asarray([nan, nan]), 0.0
        else : return asarray([nan, nan])
    elif len(points) == 1 :
        if surface : return asarray(points[0]), 0.0
        else : return asarray(points[0])

    points=asarray(points)
    Sa = 0.0
    xG, yG = 0.0, 0.0
    G = asarray([xG, yG])
    A = points[0]
    T = list(zip(points[:-1], points[1:]))+[(points[-1],points[0])]
    for b,c in T :
#    for b, c in zip(P[1:-2], P[2:-1]) :?????
        Gt = (A + b + c)/3.0
        Sat = (b[0] - A[0])*(c[1] - A[1]) - (b[1] - A[1])*(c[0] - A[0])
        G += Sat*Gt
#         debug( "pouet, a faire")
        Sa += Sat
    if Sa == 0.0 :
        if surface :  return asarray((nan, nan)), 0
        else : return asarray((nan, nan))
    else :
        if surface : return G / Sa, Sa*0.5
        else : return G / Sa

def aire(points):
    """
    Calcul de l'aire algébrique d'un polygone.
    :ATTENTION:
    - si points est un polyligne (non fermé) il est d'abord fermé
        l'aire d'un polyligne non fermé devrait être 0.0, ça n'est pas le cas ici
    - l'aire est algébrique. Donc si le polyligne se recoupe (papillote) l'aire
        peut être nulle
    Si le polygone ne se recoupe pas, (intérieur connexe), alors l'aire donne le
    sens de rotation :
    si elle est positive, ses points tournent dans le sens trigonométrique, sinon,
    le sens des aiguilles d'une montre.
    La méthode utilisée est
    - fermer le polygone : P(n)=P(0)
    - de fixer un point 'O' quelconque, ici on prend O=P[0]
    - sommer les aires des triangles A = som(A(ti)) 0<=i<n) où
    A(ti) = (1/2).OP(i)^OP(i+1) est l'aire algébrique du triangle ti=OP(i)P(i+1)
    """
    if len(points) <= 2 : return 0.0
    points=asarray(points)
    a = points[0]
    aire = 0.0
    T = list(zip(points[:-1], points[1:]))+[(points[-1],points[0])]
    for b, c in T :
        At = (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])
        aire += At
    return aire*0.5

def baryCentre(points,masses=None):
    """Retourne le barycentre du nuage de points 'points' affectés des masses 'masses' 2d"""
    if len(points)==0 :
        return None
    points=asarray(points)
#     trace('', points.shape, len(points))
    N=len(points)
    if masses is None :
        X, Y = points[:,0], points[:,1]
#         debug (X=X, Y=Y)
    else :
        masses=masses/sum(masses)
        X,Y = prod([masses,points[:,0]],axis=0), prod([masses,points[:,1]],axis=0)
    bc = [[sum(X)/N, sum(Y)/N]]
    return asarray(bc)

# def dist2(p1,p2,n=2):
#     '''retourne le carré de la distance de p1 à p2 en norme n=2'''
#     try :
# #         if isinstance(p1,QPointF) : x1,y1=p1.x(),p1.y()
# #         else :
# #         x1, y1 = p1[0], p1[1]
# #         if isinstance(p2,QPointF) : x2,y2=p2.x(),p2.y()
# #         else :
# #         x2, y2 = p2[0], p2[1]
#         return sum(vi**2 for vi in p2-p1)
# #         return (x2-x1)**2+(y2-y1)**2
#     except TypeError : # Si p1 ou p2=None
#         return nan

# def dist(p1,p2,n=2):
#     '''retourne la distance de p1 à p2 en norme n=2'''
#     return math.sqrt(dist2(p1,p2))
def normales(P, normalized=True):
    """
    Calcule les normales à un polyligne (2d) P obtenues par rotation de +pi/2 des
    vecteurs u = P[i+1]-P[i].
    :param P: ndarray((n,2),float), les points du polyligne
    :param normalized:bool, True si les vecteurs sont unitaires (divisés par leur norme)
    :returns N: ndarray((n-1,2), float), les normales extérieures(?)
    """
    X, Y = XY(diff(P))
    normales = vstack((-Y, X)).T
    if normalized :
        normales /= norm(normales, axis=1, ord=2, keepdims=True)
    return normales


def normalise2d(profil):
    """
    Attention, effet de bord ?
    Normalisation d'un polygone 2D, qui représente un profil.
    :param P:ndarray((n,2), float), les points du profil,
        avec BF=P[0]=P[-1]"""
#     debug(X=X.shape)
#     X.shape = (-1,1)
#     Y.shape = (-1,1)
#     profil = hstack((X,Y))
#     profil.shape=(-1,2)
#     profil = asarray(zip(X,Y))
#     debug(profil.tolist())
#     plt.plot(*XY(profil))
#     plt.show()
#     mexit()
    corde, nba = computeCordeAndNBA(profil)
    profil -= profil[nba]#le zero au BA
    u = profil[0]#=bf-binance_account
    alfa = atan2(u[1], u[0])
    ca, sa = cos(alfa), sin(alfa)
    X0, Y0 = XY(profil/corde)
    X0.shape = Y0.shape = (-1,1)
    return hstack((X0*ca-Y0*sa, X0*sa+Y0*ca)), corde, nba, alfa

def encombrement2D(piece):
    """
    Obsolete, conservé pour compatibilité ascendante. Utiliser plutôt encombrement(X, dim=2)
    piece doit être un np.ndarray de shape (N,2)
    retourne le parallelepipede d'encombrement d'un nuage de N points 2d de la forme (N,2)
    """
    if isinstance(piece,np.ndarray):
        points=piece.view()
        points.shape=-1,2
    #    x, y, z = X,Y,Z = points[0,:]
        Max,Min=np.max,np.min #ca reste local
        M=np.asarray([Max(points[:,0]),Max(points[:,1])])
        m=np.asarray([Min(points[:,0]),Min(points[:,1])])

        return m, M
    elif isinstance(piece,(list,tuple)) :
        #liste de points 2d [(0.0,0.0) , (1.0,0.0),...]
        xmax=xmin=piece[0][0]
        ymax=ymin=piece[0][1]
        for point in piece :
            xmax,xmin=max(xmax,point[0]),min(xmin,point[0])
            ymax,ymin=max(ymax,point[1]),min(ymin,point[1])
        return (xmin,ymin),(xmax,ymax)

def my2dPlot(XYs, legends=None, equal=False, cosmetic=None, title='No title'):
    """
    Tracer des points en 2d
    XYs est une liste de (6 maximum) tableaux de points chaque tableau est de shape (n,2) ou (n,3),
    comporte n points en 2d ou 3d. seules les deux premieres coord de chaque point sont prises en compte.
    """

    if cosmetic is None:
        cosmetic = []
    if legends is None:
        legends = []
    from matplotlib import pyplot
#    matplotlib.use('MacOSX')
    nbcourbes=len(XYs)
#     print 'nb courbes', nbcourbes
    if not cosmetic :
        cosmetic=(
                  'r-o','g-o','b-o',
                  'r-^','g-^','b-^',
                  'r:.','g:.','b:.',
                  'r*','g*','b*',
                  'r^','g^','b^',
                  'r.','g.','b.',
                  )
        cosmetic=cosmetic[:nbcourbes]
    if legends in ([],) :
        legends=[str(k) for k in range(nbcourbes)]
#     if legends is None:
#         legends = []
    xmin=ymin=10^10
    xmax=ymax=-10^10
    colors=cosmetic
    for k,xy in enumerate(XYs) :
        try: color=colors[k]
        except : color='b-'
        pyplot.plot(xy[:,0],xy[:,1],color)
        xmin=min(xmin,min(xy[:,0]))
        ymin=min(ymin,min(xy[:,1]))
        xmax=max(xmax,max(xy[:,0]))
        ymax=max(ymax,max(xy[:,1]))
    w,h=xmax-xmin,ymax-ymin
#     dizaine=int(math.log10(w))#10^dizaine <= w < 10^(1+dizaine)
    ax=pyplot.axes()
    if equal : ax.set_aspect('equal')
    ax.set_xlim(xmin-w/10,xmin+w+w/10)
    ax.set_ylim(ymin-h/10,ymin+h+h/10)
    ax.grid(which='major',axis='x',linewidth=0.75,linestyle='-',color='0.75')
    ax.grid(which='minor',axis='x',linewidth=0.25,linestyle='-',color='0.75')
    ax.grid(which='major',axis='y',linewidth=0.75,linestyle='-',color='0.75')
    ax.grid(which='minor',axis='y',linewidth=0.25,linestyle='-',color='0.75')
#    ax.xaxis.set_major_locator(pyplot.MultipleLocator(10**dizaine))
#    ax.xaxis.set_minor_locator(pyplot.MultipleLocator(10**(dizaine-1)))
#    ax.yaxis.set_major_locator(pyplot.MultipleLocator(10**dizaine))
#    ax.yaxis.set_minor_locator(pyplot.MultipleLocator(10**(dizaine-1)))
    if legends is not None:
        pylab.legend(legends,shadow=True)#, loc = (0.01, 0.55))
        ltext=pylab.gca().get_legend().get_texts()
        for k, _ in enumerate(legends) :
            pylab.setp(ltext[k],fontsize=10)#, color = 'b')

#     print ltext
    pylab.title(title)
    pyplot.show()

def XY(xy:ndarray):
    """
    J'en ai marre de taper tout le temps (pour le graphique) 'X,Y = xy[:,0],xy[:,1]'
    :param A: un tableau de points ndarray de shape (n, 2), n>=0
    :return (X, Y) : (ndarray((n,1)), ndarray((n,1))) les deux colonnes de A
    """
    if xy.shape[0] == 0 : return asarray([]),asarray([])
    return xy[:,0],xy[:,1]

def vect2d(u, v):
    """
    Retourne un réel, produit vectoriel de deux vecteurs (2d), u et v
    C'est aussi le déterminant
    """
    return u[0]*v[1] - u[1]*v[0]

det = vect2d

def rcercle(A, B, C, eps=0.0):
    """retourne le rayon dun cercle passant par les 3 points A,B,C, distincts et non alignes.
    Si les 3 points sont presque alignés retourne inf
    si deux points sont presque confondus, retourne nan"""
    A, B, C = asarray(A), asarray(B), asarray(C)
    AB, BC, CA = B-A, C-B, A-C
#     print AB, CA
    c, a, b = sqrt(AB[0]**2 + AB[1]**2), sqrt(BC[0]**2 + BC[1]**2), sqrt(CA[0]**2 + CA[1]**2),
    abc = a*b*c
    d = det(AB,CA)
    s = abs(d)
#     print s, abc
    if abc <= eps:#2 points presque confondus : infinité de cercles passant par deux points
        return nan
    elif s <= eps:# trois points presque alignés
        return inf
    else :
        return 0.5*abc/d

def rayonDeCourbure(P):
    """
    Retourne la liste des rayons de courbure du polyGONE P.
    S'il n'est pas fermé, on le ferme.
    Il y a exactement n=len(P) rayons
    """
    rayons = zeros(len(P))
    if norm(P[0]-P[-1])>1.0e-9 :#debut != fin
        rayons[ 0] = rcercle(P[-1], P[0], P[1])
        rayons[-1] = rcercle(P[-2], P[-1], P[0])
    else :
        rayons[0] = rayons[-1] = rcercle(P[-2], P[0], P[1])
    ABC = list(zip(P[0:-2], P[1:-1], P[2:]))
    for k, (A, B, C) in enumerate(ABC) :
        rayons[1+k] = rcercle(A, B, C)
#     print 'dernier', k, A, B, C, 1.0/rcercle(A, B, C)
    return rayons

def courbure(P):
    """
    Retourne la liste des courbures du polyGONE P.
    S'il n'est pas fermé, on le ferme.
    Il y a exactement n=len(P) valeurs de la courbure
    """
    courbature = zeros(len(P))
    if norm(P[0]-P[-1])>0 :#debut != fin
        courbature[ 0] = 1.0/rcercle(P[-1], P[0], P[1])
        courbature[-1] = 1.0/rcercle(P[-2], P[-1], P[0])
    else :
        courbature[0] = courbature[-1] = 1.0/rcercle(P[-2], P[0], P[1])
    ABC = list(zip(P[:-2], P[1:-1], P[2:]))
    for k, (A, B, C) in enumerate(ABC) :
        courbature[1+k] = 1.0/rcercle(A, B, C)
    return courbature

def scourbure(S, T):
    """
    ATTENTION, donne des resultats fantaisistes sur un cercle.????? a vérifier ????
    utiliser plutôt la version discrete courbure() ??????
    Parametres:
    ----------
    - S = (sx, sy) : 0<=t<=1 --> S(t)=(sx(t), sy(t) est une spline numpy,
    - T = [t0, t1, ...tn] : les valeurs du parametre t pour lesquels on calcule la courbure.
    retourne:
    --------
        un ndarray((n,)) avec les valeurs de la courbure c(ti) aux pointx sx(ti), sy(ti), ti dans T
        La courbure est l'inverse du rayon de courbure.
        Pour un arc x(t), y(t), le rayon de courbure est r(t)=((x'^2+y'^2)^3/2)/(x'y"-y'x")
        x'=x'(t), y'=y'(t) x"=x"(t), y"=y"(t).
        si ||(x",y")|| = 0, la courbure est nulle.
        cf https://fr.wikipedia.org/wiki/Rayon_de_courbure
    """
    sx, sy = S
    dx,  dy  = sx(T, 1), sy(T, 1)
    d2x, d2y = sx(T, 2), sy(T, 2)
    norm3_d2 = np.sqrt(dx**2+dy**2)**3
    # si norm_d2=0, x"(t)=y"(t)=0, c'est une droite, courbure nulle
    sc = (dx*d2y-dy*d2x) / norm3_d2
    sc[where(norm3_d2 < 1.0e-12)] = 0.0
    return sc

def simpson(f, a, b, n=10):#n doit être pair, integration precise ordre 3
    """
    Integrale de f sur [a,b], méthode de Simpson composite. (ordre 3)
    n DOIT être pair, on découpe le segment d'integration en n sous segments
    On applique la méthode de Simpson sur chaque sous segment"""
    h = float(b-a)/n
    T = linspace(a, b, n+1)
    C = f(T)
    A1 = C[0] + C[-1]
    A2 = 2*sum(C[i] for i in range(2,n) if i%2==0)
    A4 = 4*sum(C[i] for i in range(1,n) if i%2==1)
#         debug (h, A1, A2, A4, (h/3)*(A1 + A2 + A4))
    return (h/3)*(A1 + A2 + A4)

def intersectionsGridFrontiere(P, X, Y):
    """Calcule la trace de la grille cartésienne définie par X, Y sur le Polyligne P
     autrement dit les intersections de P  avec les droites
        - x = X[i] verticales et
        - y = Y[j] horizontales
        qui représentent la grille.
    :param P : de type shapely.Polygon (avec des trous eventuellement)
    #un polyligne np.ndarray de shape (n,3) ou (n,2). La 3-eme dimension est ignorée.
    :param X, Y : la grille cartésienne.
        X et Y sont des np.ndarray de shape (nx,1) et (ny,1) qui représentent
        les abscisses et les ordonnées de la grille
        - On suppose que X et Y sont croissants (i)
        - On suppose également que la grille recouvre entièrement P, et déborde, i.e.
            min(X) < xmin(P) <= xmax(P) < max(X)
            min(Y) < ymin(P) <= ymax(P) < max(Y)
    :return PD: np.ndarray((npd,2)) contenant les points
    """
    P = P.boundary
    (minx, miny, maxx, maxy) = P.bounds
    #Les numeros de droites verticales intersectant P
    iX = np.where(np.logical_and(minx<X, X<maxx))[0]
    #les droites verticales
    ms = [((X[i], Y[0]),(X[i],Y[-1]))  for i in iX]

    #Les numeros de droites horizontales intersectant P
    iY = np.where(np.logical_and(miny<Y, Y<maxy))[0]
    #Les droites horizontales concernées par P
    ms.extend([((X[0], Y[i]),(X[-1], Y[i]))  for i in iY])

    D = MultiLineString(ms)#La famille des droites de la grille

    #array_interface pour recuperer les data pures numpy
    PD = P.intersection(D)#.array_interface()
    D = [P.project(pd) for pd in PD]#abscisse curviligne des points d'intersection
    D.sort()
    PD = LineString([P.interpolate(d) for d in D])#Les points dans l'ordre
    PD = asarray(PD.xy).T
#     exit()
    #.convex_hull.exterior

#     shape = PD['shape']
    #PD.reshape(...) : si le tableau PD doit être recopié => recopie en silence
    #PD.shape=... : si le tableau PD doit être recopié => erreur
#     PD.shape = shape
    return PD

def ellipse(centre, axes, nbp=20):
    if len(centre)==2 : centre = [centre[0], centre[1], 0.0]
    a,  b= axes
    T = np.linspace(0.0, 2*pi, nbp)
    T[-1] = 0.0
    return array(list(zip(a*cos(T), b*sin(T), len(T)*[0.0])))+centre

def cercle(centre, r, nbp=20):
    return ellipse(centre, (r,r), nbp)

def plotPyVista2D(Points, connexions='auto', titre='', explain='')  -> None:
    """
    Pour tracer rapidement
    :param Points : peut-être :
        points
            • une simple liste de points [x,y,z] ou [x,y],  dim=2 ou 3
            • ou un ndarray(n,dim), dim=2 ou 3
        ou [points1, points2,...],
            • une liste de liste de points 2d ou 3d,
            • ou une liste de ndarray(n,dim), dim=2 ou 3
    :param connexions:
    :param titre:
    :param explain:
    """
    if not isinstance(Points, (list,tuple)) :
        Points = [Points]

    p = Plotter(title=titre)
    for points in Points :
        if points.shape[-1]==2 :
            points = insert(points, 2, 0.0, axis=1)
        if connexions=='auto' :
            cmesh = lines_from_points(points, close=True)
        else :
            nbn = len(connexions[0])
            connexions = [(nbn,*c) for c in connexions] #format vtk : une connexion = (2,i,j) ou (3,i,j,k)...
            #je sais pas si ca marche.comme ça..
        pmesh = PolyData(points)
        p.add_mesh(pmesh, render_points_as_spheres=True, point_size=5, color='red')#, stitle=titre)
        p.add_mesh(cmesh, render_lines_as_tubes=True, line_width=2,color='white')
    if explain : p.add_text(explain, position='upper_edge', font_size=10, font='times')
    p.show_bounds()
    p.view_xy()
    p.show()

if __name__=="__main__":
    from tests.testsgeometry2d import TestUtilitaires2d
    t = TestUtilitaires2d()
    if 1 : t.testIsInside()
    if 1 : t.testPointsDoubles()
    if 1 : t.testSpline2D()
    if 1 : t.testRayonCercle()
    if 1 : t.testSCourbure()
    if 1 : t.testSegmentPlusProche()
    if 1 : t.testSegmentPlusProche1()
    if 1 : t.testSimpson()
    if 1 : t.testTrou()

#     import cProfile, pstats
#     filename = OUTPUT,'profile.stats')
#     cProfile.run('main()',filename)
#     stats=pstats.Stats(filename)
#     stats.strip_dirs()
#     stats.sort_stats('cumulative')
#     stats.sort_stats('time')
#     stats.print_stats()




