import numpy as np
import math 
from scipy.interpolate import splprep, splev

class FacialGeometricIndicators:
    def __init__(self, landmarks):
        """
        Initialize with 68 facial landmarks.
        landmarks: List of tuples (x, y) representing the facial landmarks.
        """
        self.landmarks = landmarks
        self.fgi_values = {}

    def calculate_distance_x(self, idx):

        point1 = self.landmarks[idx[0]][0]
        point2 = self.landmarks[idx[1]][0]
        return abs(point1 - point2)
    
    def calculate_distance_y(self, idx):

        point1 = self.landmarks[idx[0]][1]
        point2 = self.landmarks[idx[1]][1]
        return abs(point1 - point2)
    
    def polygon_area(self, indices):
        """Calculates the area of a polygon whose vertices are given by the indices list using the Shoelace formula."""
        n = len(indices)
        area = 0
        for i in range(n):
            x1, y1 = self.landmarks[indices[i]]
            x2, y2 = self.landmarks[indices[(i + 1) % n]]  # Wrap around using modulo
            area += x1 * y2 - y1 * x2
        return abs(area) / 2
    
    def compute_curvature(self, points):

        points = np.array([self.landmarks[idx] for idx in points])
        tck, u = splprep(points.T, s=0)
        der1 = splev(u, tck, der=1)  
        der2 = splev(u, tck, der=2) 
        curvature = np.abs(der1[0]*der2[1] - der1[1]*der2[0]) / np.power(der1[0]**2 + der1[1]**2, 1.5)
        return curvature
    
    # def plot_curve(self, tck):
    #     unew = np.linspace(0, 1, 100)
    #     out = splev(unew, tck)
    #     plt.figure()
    #     plt.plot(out[0], out[1], 'b-', self.points[:, 0], self.points[:, 1], 'ro')
    #     plt.show()

    def angle_between_points(self, p1, p2, p3):
        def vector(a, b):
            return (b[0] - a[0], b[1] - a[1])

        v1 = vector(p2, p1)
        v2 = vector(p2, p3)

        # Calculate the angle in radians between the two vectors
        angle_rad = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
        angle_rad = abs(angle_rad)  # Absolute value to get the positive angle

        # Normalize angle to [0, pi]
        if angle_rad > math.pi:
            angle_rad -= 2 * math.pi
        elif angle_rad < -math.pi:
            angle_rad += 2 * math.pi

        return math.degrees(abs(angle_rad))

    def calculate_all(self):
        """
        Calculate all Facial Geometric Indicators based on the provided formulas.
        """

        # # Left eyebrow width 
        # self.fgi_values['LeftEyebrowWidth'] = self.calculate_distance_x((21, 17))

        # # Right eyebrow width 
        # self.fgi_values['RightEyebrowWidth'] = self.calculate_distance_x((26, 22))

        # Left eye width 
        self.fgi_values['LeftEyeWidth'] = self.calculate_distance_x((39, 36))

        # Right eye width 
        self.fgi_values['RightEyeWidth'] = self.calculate_distance_x((45, 42))

        # Eye width
        self.fgi_values['AvgEyeWidth'] = (self.fgi_values['LeftEyeWidth'] + self.fgi_values['RightEyeWidth'])/2

        # Left eye breadth
        self.fgi_values['LeftEyeBreadth'] = (self.calculate_distance_y((37, 41)) + self.calculate_distance_y((38, 40)))/2

        # Right eye breadth 
        self.fgi_values['RightEyeBreadth'] = (self.calculate_distance_y((43, 47)) + self.calculate_distance_y((44,46)))/2

        # Eye breath
        self.fgi_values['AvgEyeBreadth'] = (self.fgi_values['LeftEyeBreadth'] + self.fgi_values['RightEyeBreadth'])/2

        # Mouth width
        self.fgi_values['MouthWidth'] = self.calculate_distance_x((60, 64))

        # Upper mouth thickness
        self.fgi_values['UpperLipThickness'] = (self.calculate_distance_y((50,61)) + self.calculate_distance_y((52,63)))/2

        # Lower mouth thickness
        self.fgi_values['LowerLipThickness'] = (self.calculate_distance_y((67, 58)) + self.calculate_distance_y((66,57)) + self.calculate_distance_y((65,56)))/3

        # Mouth shape
        self.fgi_values['MouthShape'] = (self.fgi_values['UpperLipThickness'] + self.fgi_values['LowerLipThickness'])/self.fgi_values['MouthWidth']
        
        # Ratio of mouth width to face width
        LeftFaceWidth = (self.calculate_distance_x((1,48)) + self.calculate_distance_x((2,48)) + self.calculate_distance_x((3,48)) + self.calculate_distance_x((4,48)))/4
        RightFaceWidth = (self.calculate_distance_x((12,54)) + self.calculate_distance_x((13,54)) + self.calculate_distance_x((14,54)) + self.calculate_distance_x((15,54)))/4
        self.fgi_values['RmouthW2faceW'] = self.fgi_values['MouthWidth']/(LeftFaceWidth+RightFaceWidth+self.fgi_values['MouthWidth'])

        # Ratio of facial features area to face area
        self.fgi_values['RFeaturesA2WholeA'] = (self.polygon_area((36, 37, 38, 43, 44, 45, 35, 31)) + self.polygon_area((31, 35, 54, 57, 48))) / self.polygon_area(list(range(0,17))+list(range(26,16,-1)))

        # Smile arc
        self.fgi_values['SmileArc'] = np.mean(self.compute_curvature([60, 59, 58, 57, 56, 55, 64]))

        # Jaw line angle
        self.fgi_values['JawLineAngle'] = (self.angle_between_points(self.landmarks[0], self.landmarks[4], self.landmarks[8]) + self.angle_between_points(self.landmarks[16], self.landmarks[12], self.landmarks[8]))/2

        # Jaw curvature
        self.fgi_values['JawCurvature'] = (np.max(self.compute_curvature(list(range(2,8)))) + np.max(self.compute_curvature(list(range(9,15)))))/2
        return self.fgi_values

    def get_fgi_values(self):
        """
        Get the calculated FGI values.
        """
        return self.fgi_values if self.fgi_values else self.calculate_all()
    