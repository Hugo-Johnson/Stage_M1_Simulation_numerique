#%%
"""
Source : Monte_carlo_bj_thesis.py by @benoi


This example demonstrates the evolution of a cloud
of particles subjected to crossed magnetic and electric field.

https://github.com/cchandre/pyhamsys/blob/main/README.md
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import pyqtgraph.exporters

pg.setConfigOption('background', 'w')   # white background
pg.setConfigOption('foreground', 'k')   # black axes, labels, etc.
pg.setConfigOptions(antialias=False)

import pyhamsys
from pyhamsys import HamSys, Parameters

# Variables --------------------------------------------------------------------------------
B=0.7        # magnetic field (a.u.)
vd= 5.0     #drif velocity

E_opt= 5    # energy of the optical phonons)
kt=8.0      #  temperature of the initial particle distribution
vf=8.0      # Fermi velocity (a.u.) for graphene

# E = 10      # Test Electric field 
T = 0 # time compteur
## Parameters ------------------------------------------------------------------------
step =  0.1* 2 * np.pi / 50		# integration timestep 
N = 500	                    # number of trajectories
"""
Y_dot renvoie toujours le problème Hamiltonien "H" sous la forme:
y0 = d_{py}H
y1 = - d_{px}H
"""
def y_dot_G0(y):   # for graphene
    q, p = np.split(y, 2)
    y0=    p * (1+q*q)/2+ vd
    y1=   -q*  (1+p*p)/2
    return np.concatenate((y0, y1), axis=None)


def y_dot_G(y):   # for graphene
    q, p = np.split(y, 2)
    y0 =  vf*  p / np.sqrt(0.1+p*p+ q*q)+ vd
    y1 = -vf*  q / np.sqrt(0.1+p*p+ q*q)
    return np.concatenate((y0, y1), axis=None)

def y_dot_P(y):  # for parabolic band
    q, p = np.split(y, 2)
    y0 =  2*  p + vd
    y1 = -2*  q 
    return np.concatenate((y0, y1), axis=None)


def y_dot_BHZ(y):
    
    scale = 1000.0
    M = 13.0/scale
    # B = -1990.0/scale*0
    D = 810/scale*1
    A = 353.0/scale*0
    q, p = np.split(y, 2)
    r1= (M-B*(q*q+p*p))
    denom= np.sqrt(r1*r1+ A*A*(p*p+q*q))
    y0 = -   2*D*p + (-2*B*p*r1 + A*A*2*p) / denom + vd
    y1 = - (-2*D*q + (-2*B*q*r1 + A*A*2*q) / denom)
    return np.concatenate((y0, y1), axis=None)

# ------ Bande Fermions de Dirac, papier Sergei ------------------------------


def y_dot_Hugo(y):

    def varepsilon(p):
        M = 1
        px, py = np.split(p, 2)
        return np.sqrt(M**2 +vf** 2 * (px** 2 + py** 2))
    e = 1
    c = 1
   
    G1 = ( e * B * vf**2) / (c * varepsilon(y)) 
    G2 = (e * B / c)
    q, p = np.split(y, 2)
    y0 = G2 * ((vf**2 * p / varepsilon(y)) + vd) 
    y1 = -G1 * q

    return np.concatenate((y0, y1), axis=None)

 #-------------- Bande Parabolique -------------------------
def y_dot_Hugo_Parabolic(y): #resolution papier pour bande parabolique
    e = 1
    c = 1
    alpha = 2.5
    G =  e * B  / c

    q, p = np.split(y, 2)
    y0 = G * (alpha * p + vd)
    y1 = -G * alpha * q
    return np.concatenate((y0, y1), axis=None)
# ---------------- Bande linéaire ---------------------------
def y_dot_Hugo_lineaire(P): #resolution papier pour bande linéaire
    e, c = 1, 1
    vF = 0.8
    G =  e * B  / c
    q, p = np.split(P, 2)
    def varepsilon_graphene(p):
        px = p[0]
        py = p[1]
        return  vF * np.sqrt((px**2 + py**2)) #♦+ 100
    y0 = G * ((vF**2 * p / varepsilon_graphene(P)) + vd)
    y1 = -G * (vF**2 * q / varepsilon_graphene(P))
    return np.concatenate((y0 , y1), axis=None)

# --------------------------------------------------

y_dot = y_dot_G0 #select y_dot

J20 = np.array([[1, 1], [1, 1]]) / 2
J22 = np.array([[1, -1], [-1, 1]]) / 2
J40 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]) / 2
J42c = np.array([[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]]) / 2
J42s = np.array([[0, -1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0]]) / 2

def coupling(h:float) -> np.ndarray:
    omega=10 # Coupling parameter in the extended phase space (see [1] of HamSys)
    return J40 + np.cos(2 * omega * h) * J42c + np.sin(2 * omega * h) * J42s

def chi(h:float, t, y:np.ndarray) -> np.ndarray:
	y_ = np.split(y, 2,)
	y_[1] += h * y_dot(y_[0])
	y_[0] += h * y_dot(y_[1])
	yr = np.concatenate((y_[0], y_[1]), axis=None)
	yr = np.einsum('ij,j...->i...',
                coupling(h),
                np.split(yr,  4)
                ).flatten()
	return yr

def chi_star(h:float, t,  y:np.ndarray) -> np.ndarray:
	y_ = np.split(y, 4, )
	yr = y_ 
	yr = np.einsum('ij,j...->i...', coupling(h), yr).flatten()
	y_ = np.split(yr, 2)
	y_[0] += h * y_dot(y_[1])
	y_[1] += h * y_dot(y_[0])
	return np.concatenate([_ for _ in y_], axis=None)

def integrate( y_:np.ndarray) ->  np.ndarray :
    sol = pyhamsys.solve_ivp_symp(chi, chi_star, (0, step), y_, params=Parameters(step=step))
    return sol.y[:, 1]

## Initial conditions
def init_distribution():
    
    en= np.random.default_rng().exponential(scale=kt, size=N)
    p=np.sqrt(en)
    phi=2*np.pi*np.random.default_rng().uniform(size=N)

    px= np.cos(phi)*p
    py= np.sin(phi)*p
    y0 = np.concatenate((px, py), axis=None)
    return (px, py, y0)

def reset_distribution():
    global px, py, y0, y_, ptr, T
    ptr = 0
    T = 0
    (px, py, y0)= init_distribution()
    y_ = np.tile(y0, 2).astype(np.float64)


app = pg.mkQApp("Streaming Example")

def vd_changed():
    global vd
    vd = widget.value()
    # print(vd)

def B_changed():
     global B
     B = widget_Magnetic.value()

def band(name):
     global y_dot
     y_dot = y_dot_options[name]

# def E_changed():
#     global E
#     E = widget_Electric.value()

y_dot_options = {
    'Graphène 0': y_dot_G0,
    'Graphène 1': y_dot_G,
    'Parabolic': y_dot_P,
    'BZH' : y_dot_BHZ,
    'Hugo (Bande dispersion of massive Dirac fermions) ' : y_dot_Hugo,
    "Hugo (Parabolique)" : y_dot_Hugo_Parabolic,
    "Hugo (lineaire, graphène)" : y_dot_Hugo_lineaire
}
material_widget = QtWidgets.QComboBox()
for name in y_dot_options:
    material_widget.addItem(name)
material_widget.currentTextChanged.connect(band)


widget = QtWidgets.QDoubleSpinBox()
widget.setRange(0, 10)
widget.setSingleStep(0.2)
widget.setValue(vd)
widget.valueChanged.connect(vd_changed)

widget_Magnetic = QtWidgets.QDoubleSpinBox()
widget_Magnetic.setRange(0.1, 10)
widget_Magnetic.setSingleStep(0.1)
widget_Magnetic.setValue(B)
widget_Magnetic.valueChanged.connect(B_changed)

# widget_Electric = QtWidgets.QDoubleSpinBox()
# widget_Electric.setRange(0, 100)
# widget_Electric.setSingleStep(10)
# widget_Electric.setValue(E)
# widget_Electric.valueChanged.connect(B_changed)



reset_dist= QtWidgets.QPushButton()
reset_dist.setText('reset')
reset_dist.clicked.connect(reset_distribution)
# - Label ------------------------
label = QtWidgets.QLabel()
label.setText('v_drift:')
label_Magnetic = QtWidgets.QLabel()
label_Magnetic.setText('B_force')
# label_Electric = QtWidgets.QLabel()
# label_Electric.setText('E_force')

# --------------------------------
rcheck = QtWidgets.QCheckBox('optical')
rcheck.setChecked(True)
lcheck = QtWidgets.QCheckBox('plot local')

p6 = pg.PlotWidget(background='black')
#p6 = pg.plot()
p6.setAspectLocked(True)

p6.enableAutoRange('xy', False)

# ---- Axis limits ----
p6.setXRange(-E_opt*1.01, E_opt*1.01)
p6.setYRange(-E_opt*1.01, E_opt*1.01)

layout = pg.LayoutWidget()
layout.addWidget(label,row=0, col=0)
layout.addWidget(widget, row=0, col=1)
layout.addWidget(label_Magnetic, row=1, col=0)
layout.addWidget(widget_Magnetic, row=1, col=1)
layout.addWidget(material_widget, row=0, col=2)
layout.addWidget(rcheck, row=0, col=3)
layout.addWidget(reset_dist,row=0, col=4)
# layout.addWidget(label_Electric, row=2, col=0)
# layout.addWidget(widget_Electric, row=2, col=1)

layout.addWidget(p6, row=3, col=0, colspan=5)
layout.resize(900,600)
layout.show()

#xopt = E_opt*np.cos(np.linspace(0, 2*np.pi, 1000))
#yopt = E_opt*np.sin(np.linspace(0, 2*np.pi, 1000))
#p6.plot(xopt, yopt)

# ---- Draw circle ----
circle = QtWidgets.QGraphicsEllipseItem(-E_opt, -E_opt, 2*E_opt, 2*E_opt)
circle.setPen(pg.mkPen(color=(150,150,150), width=4))  # thick grey
p6.addItem(circle)

# ---- Draw axes ----
x_axis = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('k', width=2))
y_axis = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('k', width=2))
p6.addItem(x_axis)
p6.addItem(y_axis)


text = pg.TextItem('B= %0.1f, vd= %0.1f, t_step= %0.1f' % (B, vd,0))
text.setFont(QtGui.QFont("Arial", 14))   # <-- correct import
#p6.addItem(text)

text.setParentItem(p6.getViewBox())   # <-- prevents shifting
text.setPos(-5, 5.)

p6.setLabel('left', "p_y")
p6.setLabel('bottom', "p_x")

(px, py, y0)= init_distribution()

curve = p6.plot( 
    px,py, 
    pen=None,
    symbol='o', 
    symbolPen=None, 
    symbolSize=5,
    symbolBrush=(30, 144, 255, 150) 
    )

#exporter = pg.exporters.ImageExporter(p6.scene())

ptr = 0

#the explicit resolution of nonseparable Hamiltonian 
#requires the uses of an augmented (x2) Hamiltonian.
y_ =  np.tile(y0, 2).astype(np.float64)

def update():
    global curve,  ptr, p6, y_, T, step
    E_init= 0.2

    y_ = integrate( y_)
    y = np.split(y_,  4, )
    
    y_qp = np.concatenate(((y[0] + y[2]) / 2, (y[1] + y[3]) / 2), axis=0)
    (px,py) = np.split(y_qp,2)
    
    if rcheck.isChecked():
        mask =np.sqrt(px*px+py*py)> E_opt
        px_0 = np.random.normal(scale = E_init,size=(N))
        py_0 = np.random.normal(scale= E_init,size=(N))
        px[mask]= px_0[mask] 
        py[mask]= py_0[mask] 
        
        y_t = np.concatenate((px, py), axis=None)
        y_ = np.tile(y_t, 2).astype(np.float64)
        
    curve.setData(px,py)
    text.setText('B= %0.1f, vd= %0.1f, t_step= %0.1f, TIME= %0.1f' % (B, vd, ptr, T))
    if ptr == 0:
        p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    ptr += 1
    T = ptr * step
    #for export: uncheck the following line
    # exporter.export('scenes_P11/1_scene_'+str(ptr).zfill(6)+'.png')
    #then
    #ffmpeg -i 1_scene_%06d.png -c:v libx264 -r 30 final.mp4
    #or choose encoder mpeg for powerpoint...
    # ffmpeg -i 1_scene_%06d.png -vf scale=1080:-2 -c:v mpeg4 -r 100 dirac_2_mpeg4.mp4
    

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(60)

if __name__ == '__main__':
    pg.exec()
# %%
