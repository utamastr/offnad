import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, timezone
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs

def quatmultiply(q1, q2):
    x1, y1, z1, s1 = q1
    x2, y2, z2, s2 = q2
    x = s1 * x2 + x1 * s2 + y1 * z2 - z1 * y2
    y = s1 * y2 + y1 * s2 + z1 * x2 - x1 * z2
    z = s1 * z2 + z1 * s2 + x1 * y2 - y1 * x2
    s = s1 * s2 - x1 * x2 - y1 * y2 - z1 * z2
    magnitude = np.sqrt(x**2 + y**2 + z**2 + s**2)
    if magnitude > 0:
        x /= magnitude
        y /= magnitude
        z /= magnitude
        s /= magnitude
    return [x, y, z, s]

def quatconj(quaternion):
    quaternion_ = [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]
    return quaternion_

def quatrotate(quaternion, vector):
    new_vector = quatmultiply(quatmultiply(quaternion, [vector[0], vector[1], vector[2], 0]), quatconj(quaternion))
    return new_vector[0:3]

def ypr_from_eci(t_s, q, lon, lat, v):
    t = datetime.strptime(t_s, '%Y/%m/%d %H:%M:%S.%f')
    tn = ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second)
    gmst_rad = tn.gmst/24*np.deg2rad(360)

    q10 = np.cos((np.deg2rad(lon) + gmst_rad)/2)
    q11 = 0
    q12 = 0
    q13 = np.sin((np.deg2rad(lon) + gmst_rad)/2)
    q1 = [q11, q12, q13, q10]

    q20 = np.cos((-1.570796326794897-np.deg2rad(lat))/2)
    q21 = 0
    q22 = np.sin((-1.570796326794897-np.deg2rad(lat))/2)
    q23 = 0
    q2 = [q21, q22, q23, q20]

    q_eci_ned = quatmultiply(q1, q2)
    q_ned_eci = quatconj(q_eci_ned)

    q_eci_sat = q
    q_ned_sat = quatmultiply(q_ned_eci, q_eci_sat)
    
    norm_v = np.linalg.norm(v, axis=0)
    v_eci_sat = v/norm_v
    v_ned_sat = quatrotate(q_ned_eci, v_eci_sat)
    az_v = np.arccos(v_ned_sat[0])
    

    q_nad_ned = ypr2q(az_v, 0, 0)
    q_nad_sat = quatmultiply(q_nad_ned, q_ned_sat)
    y, p, r = quaternion2ypr(q_nad_sat)
    ypr = [y, p, r]
    
    return q_nad_sat, ypr

def ypr2q(yaw, pitch, roll):
    q_z = [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]
    q_y = [0, np.sin(pitch / 2), 0, np.cos(pitch / 2)]
    q_x = [np.sin(roll / 2), 0, 0, np.cos(roll / 2)]
    q = quatmultiply(q_z, quatmultiply(q_y, q_x))
    magnitude = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    q /= magnitude
    return q

def quaternion2ypr(q):
    magnitude = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    q /= magnitude
    x, y, z, s = q

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (s * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (s * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Roll (X-axis rotation)
    sinr_cosp = 2 * (s * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    yaw = np.degrees(yaw)
    pitch = np.degrees(pitch)
    roll = np.degrees(roll)
    return yaw, pitch, roll

df = pd.read_csv('quat_20181009T015005_20181009T015055.txt', delimiter=';')
eph = load('de421.bsp')
earth, sun = eph['earth'], eph['sun']

t = [datetime.strptime(t, '%Y/%m/%d %H:%M:%S.%f') for t in df['System Time']]
ts = load.timescale()
line1 = '1 41603U 16040E   18282.19159434  .00000031  00000-0  46613-5 0  9991'
line2 = '2 41603  97.3998 341.5659 0012935 196.0772 252.5874 15.19613888127391'

satellite = EarthSatellite(line1, line2, 'LAPAN-A3', ts)

rx = []
ry = []
rz = []
vx = []
vy = []
vz = []
lat = []
lon = []
alt = []


for ti in t:
    tn = ts.utc(ti.year, ti.month, ti.day, ti.hour, ti.minute, ti.second)
    geocentric = satellite.at(tn)
    r1, r2, r3 = geocentric.frame_xyz(itrs).km
    rx.append(r1)
    ry.append(r2)
    rz.append(r3)
    v1, v2, v3 = geocentric.velocity.km_per_s
    vx.append(v1)
    vy.append(v2)
    vz.append(v3)
    
    r = wgs84.geographic_position_of(geocentric)
    lat.append(r.latitude.degrees)
    lon.append(r.longitude.degrees)
    alt.append(r.elevation.km)
    
df_o = pd.DataFrame({'rx_eci' : rx,
                     'ry_eci' : ry,
                     'rz_eci' : rz,
                     'vx' : vx,
                     'vy' : vy,
                     'vz' : vz,
                     'lat' : lat,
                     'lon' : lon,
                     'alt' : alt})

yaw = []
pitch = []
roll = []
i = 0
while i < len(df):
    q = quatmultiply([df['q1 (STS B)'][i], df['q2 (STS B)'][i], df['q3 (STS B)'][i], df['q0 (STS B)'][i]], [0, 0, 1, 0])
    v = [df_o['vx'][i], df_o['vy'][i], df_o['vz'][i]]
    q_nad_sat, ypr = ypr_from_eci(df['System Time'][i], q, df_o['lon'][i], df_o['lat'][i], v)
    yaw.append(ypr[0])
    pitch.append(ypr[1])
    roll.append(ypr[2])
    i += 1

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, yaw, label='yaw')
ax.plot(t, pitch, label='pitch')
ax.plot(t, roll, label='roll')
ax.grid(True)
ax.set_xlabel('Time')
ax.set_ylabel('Angle ($^{o}$)')
ax.legend()