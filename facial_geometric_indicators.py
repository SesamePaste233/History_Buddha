class FacialGeometricIndicators:
    def __init__(self, landmarks):
        """
        Initialize with 68 facial landmarks.
        landmarks: List of tuples (x, y) representing the facial landmarks.
        """
        self.landmarks = landmarks
        self.fgi_values = {}

    def calculate_distance_x(self, idx):

        point1 = self.landmarks[idx[0]-1][0]
        point2 = self.landmarks[idx[1]-1][0]
        return point1 - point2
    
    def calculate_distance_y(self, idx):

        point1 = self.landmarks[idx[0]-1][1]
        point2 = self.landmarks[idx[1]-1][1]
        return point1 - point2

    def calculate_all(self):
        """
        Calculate all Facial Geometric Indicators based on the provided formulas.
        """
        # Eye height (A)
        self.fgi_values['EyeHeight'] = sum(self.calculate_distance_y(idx) for idx in [(42,38), (41,39), (47,45), (48,44)]) / 4

        # Eyebrow to upper eyelid (B)
        self.fgi_values['EyebrowToUpperEyelid'] = sum(self.calculate_distance_y(idx) for idx in [(38, 20), (40, 22), (45, 25), (43, 23)]) / 4

        # Eye to nose bottom (C)
        self.fgi_values['EyeToNoseBottom'] = sum(self.calculate_distance_y(idx) for idx in [(33, 40), (35, 41), (33, 43), (35, 48)]) / 4

        # Nose bottom to chin (D)
        self.fgi_values['NoseBottomToChin'] = sum(self.calculate_distance_y(idx) for idx in [(7, 32), (11, 36)]) / 2

        # Right cheek width (E)
        self.fgi_values['RightCheekWidth'] = self.calculate_distance_x((17, 9))

        # Left cheek width (F)
        self.fgi_values['LeftCheekWidth'] = self.calculate_distance_x((9, 1))

        # Right chin width (G)
        self.fgi_values['RightChinWidth'] = self.calculate_distance_x((11, 9))

        # Left chin width (H)
        self.fgi_values['LeftChinWidth'] = self.calculate_distance_x((9, 7))

        # Left eyebrow width (I)
        self.fgi_values['LeftEyebrowWidth'] = self.calculate_distance_x((22, 18))

        # Right eyebrow width (J)
        self.fgi_values['RightEyebrowWidth'] = self.calculate_distance_x((27, 23))

        # Left eye width (K)
        self.fgi_values['LeftEyeWidth'] = self.calculate_distance_x((40, 37))

        # Right eye width (L)
        self.fgi_values['RightEyeWidth'] = self.calculate_distance_x((46, 43))

        # Left mouth width (M)
        self.fgi_values['LeftMouthWidth'] = self.calculate_distance_x((52, 49))

        # Right mouth width (N)
        self.fgi_values['RightMouthWidth'] = self.calculate_distance_x((55, 52))

        # Left nose width (O)
        self.fgi_values['LeftNoseWidth'] = self.calculate_distance_x((34, 32))

        # Right nose width (P)
        self.fgi_values['RightNoseWidth'] = self.calculate_distance_x((36, 34))

        return self.fgi_values

    def get_fgi_values(self):
        """
        Get the calculated FGI values.
        """
        return self.fgi_values if self.fgi_values else self.calculate_all()
    
    def calculate_fsa(self):
        E = self.fgi_values['RightCheekWidth']
        F = self.fgi_values['LeftCheekWidth']
        G = self.fgi_values['RightChinWidth']
        H = self.fgi_values['LeftChinWidth']
        I = self.fgi_values['LeftEyebrowWidth']
        J = self.fgi_values['RightEyebrowWidth']
        K = self.fgi_values['LeftEyeWidth']
        L = self.fgi_values['RightEyeWidth']
        M = self.fgi_values['LeftMouthWidth']
        N = self.fgi_values['RightMouthWidth']
        O = self.fgi_values['LeftNoseWidth']
        P = self.fgi_values['RightNoseWidth']

        # Calculate FSA according to the formula provided
        FSA = (abs(E/F - 1) + abs(G/H - 1) + abs(I/J - 1) + abs(K/L - 1) + abs(M/N - 1) + abs(O/P - 1)) / 6

        # Add FSA to the fgi_values dictionary
        self.fgi_values['FSA'] = FSA

        return FSA