import numpy as np

class Fibril:

    def __init__(self, init_center, direction, radius):

        # Set the initial center of the fibril 
        self.init_center = init_center
        self.center = self.init_center

        # Set the direction of the fibril
        self.direction = direction

        # Set the radius of the fibril
        self.radius = radius

        # Initialize the fibril length to 0
        self.length = 0

        # Set fibril orientation to fibril direction
        orientation = self.direction
        self.orientation_theta = np.arccos(orientation[2])
        self.orientation_psi   = np.arctan2(orientation[1], orientation[0])

        # Map orientation angles to rgb colorspace
        r = np.sin(self.orientation_theta)*np.cos(self.orientation_psi)
        g = np.sin(self.orientation_theta)*np.sin(self.orientation_psi)
        b = np.cos(self.orientation_theta)
        self.color = np.array([r, g, b])
        
        self.fibril_indices = None
    
    @staticmethod
    def generate_sphere_mask(radius:int) -> np.ndarray:
        """
        Generate a 3D binary mask representing a solid sphere within a cube-like space.

        Args:
            radius (int): The radius of the sphere to be generated.

        Returns:
            np.ndarray: A 3D NumPy array (binary mask) where the elements are `True` (1) 
            inside the sphere and `False` (0) outside of it.

        The function creates a binary mask in the form of a cube with side lengths of 
        `2 * radius + 1`. The mask has a solid sphere centered in it with the specified 
        `radius`. The values inside the sphere are set to `True`, and those outside the 
        sphere are set to `False`.

        """
        
        mask_size = 2 * radius + 1
        center = radius

        # Create coordinate grids for x, y, and z dimensions within the cube
        x, y, z = np.ogrid[:mask_size, :mask_size, :mask_size]

        # Calculate the squared distance from the center of the sphere/cube.
        distance_sq = (x - center)**2 + (y - center)**2 + (z - center)**2

        # Create a binary mask by comparing the squared distances to the squared radius.
        mask = distance_sq <= radius**2
        return mask

    @staticmethod
    def apply_sphere_mask(array:np.ndarray, mask:np.ndarray, center_position, allow_overlap=True, periodic_bc=True) -> tuple:
        """
        Apply a 3D binary mask (sphere) to a 3D NumPy array.

        Args:
            array (np.ndarray): A 3D NumPy array to which the mask is applied.
            mask (np.ndarray): A 3D binary mask (binary array) representing the shape to be applied to the `array`.
            center_position (tuple or list): A 3-element tuple or list specifying the center position for applying the mask.
            allow_overlap (bool, optional): A flag indicating whether overlap between the mask and array is allowed (default: True).
            periodic_bc (bool, optional): A flag indicating whether to apply periodic boundary conditions (default: True).

        Returns:
            tuple or None: If successful, returns a tuple of modified indices where the mask has been applied in the `array`. If unsuccessful (e.g., due to overlap or out-of-bounds), returns None.

        This function applies a 3D binary mask (sphere) to a 3D array, centered at the specified `center_position`. The function offers options to handle overlap and periodic boundary conditions.

        If `allow_overlap` is True, the mask is applied without checking for overlap with existing values in the array. If `allow_overlap` is False, the function checks for overlap and returns None if overlap is detected.

        If `periodic_bc` is True, the function applies periodic boundary conditions, allowing the mask to wrap around the array's boundaries. If `periodic_bc` is False, the function checks if the mask extends beyond the array's boundaries and returns None if it does.
        """

        array_shape = np.array(array.shape)
        mask_shape  = np.array(mask.shape)

        # Broadcast arrays
        center_position_br = center_position[:, np.newaxis, np.newaxis, np.newaxis]
        mask_shape_br = mask_shape[:, np.newaxis, np.newaxis, np.newaxis]
        array_shape_br = array_shape[:, np.newaxis, np.newaxis, np.newaxis]

        # Get valid indices of mask and array
        mask_indices = np.indices(np.array(mask.shape))
        valid_indices = mask_indices + center_position_br
        valid_indices -= mask_shape_br // 2 

        # Apply periodic boundary conditions
        if periodic_bc:
            # valid_indices = [valid_indices[dim] % array_shape_br[dim] for dim in range(3)]
            for dim in range(3):
                valid_indices[dim] = valid_indices[dim] % array_shape_br[dim]

        # Check if array is out of bounds
        out_of_bounds = np.any(valid_indices >= array_shape_br) or np.any(valid_indices < 0)
        if out_of_bounds:
            return None
        
        array_values = array[valid_indices[0], valid_indices[1], valid_indices[2]]
        mask_values = mask[mask_indices[0], mask_indices[1], mask_indices[2]]

        # Check if arrays overlap
        if allow_overlap == False:
            overlap =  np.any(array_values & mask_values)
            if overlap == True:
                return None
        
        modified_indices = np.where(~array_values & mask_values)
        modified_indices = modified_indices + center_position.reshape(3, 1) - mask_shape.reshape(3,1)//2
        for dim in range(3):
            modified_indices[dim] = modified_indices[dim] % array_shape[dim]

        return tuple(modified_indices)

    @staticmethod
    def concatenate_indices(indices1, indices2):
        return tuple(np.hstack((a,b)).astype(int) for a,b in zip(indices1, indices2))

    def calculate_fibril_length(self, left_center:np.ndarray, right_center:np.ndarray) -> float:
        return np.linalg.norm(right_center - left_center) + 2*self.radius

    def grow_fibril_along_direction(self, box: np.ndarray, length_range=[0, 40], periodic_bc=True, symmetrical_growth=False):
        
        assert(len(length_range) == 2)
        min_length, max_length = length_range

        # Boolean mask of sphere with a given radius
        mask = Fibril.generate_sphere_mask(self.radius)

        # Boolean mask of the current growing fibril
        fibril_array = np.zeros(box.shape, dtype=np.bool_)

        # Indices of (x,y,z) coordinates of each fibril voxel
        fibril_indices = (np.array([]), np.array([]), np.array([]))
        
        # Flags for allowing left and right fibril growth
        grow_left  = True
        grow_right = True

        left_center = right_center = self.center

        # Nested function for growing the fibril in a given direction
        def _grow_fibril(can_grow:bool, growth_direction:int, iteration:int) -> (bool, np.ndarray):

            nonlocal fibril_indices

            growth_factor = self.radius * (1 - 0.9)
            current_center = np.round(self.center + growth_direction * growth_factor * (iteration) * self.direction).astype(int)
            new_center = np.round(self.center + growth_direction * growth_factor * (iteration + 1) * self.direction).astype(int)

            # Get the indices of new fibril growth
            modified_indices = Fibril.apply_sphere_mask(fibril_array, mask, new_center, allow_overlap=True, periodic_bc=periodic_bc)

            # Check if the fibril growth is invalid or intersects with other fibrils
            new_growth_intersects = modified_indices is None or not np.all(~box[modified_indices])
            if new_growth_intersects:
                return False, current_center

            fibril_indices = Fibril.concatenate_indices(fibril_indices, modified_indices)
            fibril_array[modified_indices] = True

            return True, new_center

        max_iterations = 10_000
        for i in range(max_iterations):
            
            # Stop growth if the fibril length exceeds the max length
            fibril_length = self.calculate_fibril_length(left_center, right_center)
            if fibril_length > max_length:
                grow_left = False
                grow_right = False
            
            # Grow fibrils if possible
            if grow_left:
                grow_left, left_center = _grow_fibril(grow_left, -1, i)
            if grow_right:
                grow_right, right_center = _grow_fibril(grow_right, 1, i)


            # Stop all fibril growth if fibril can only grow symmetrically
            any_side_stopped = not grow_left or not grow_right
            if symmetrical_growth and any_side_stopped:
                grow_left = False
                grow_right = False
            
            both_sides_stopped = not grow_left and not grow_right
            if both_sides_stopped:
                self.length = self.calculate_fibril_length(left_center, right_center)
                self.fibril_indices = fibril_indices

                # Check if the fibril grew to minimum length
                if self.length < min_length:
                    return False
                return True