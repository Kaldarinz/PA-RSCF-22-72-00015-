    def get_raw_points(self, add_zero_points: bool=False) -> np.ndarray:
        """
        2D array containing deep copy of measured points.

        The first index is horizontal axis, the second is vertical.
        Indexing starts ar left bottom corner of scan.
        As amount of points along fast scan axis can vary from line
        to line, size of the array in this direction corresponds to
        the longest scan line in ``data``. Size along slow scan axis
        is ``spoints``. Default value of not scanned points is 
        ``None``.\n

        Attributes
        ----------
        `add_zero_points` if `True` duplicate initial scan points so that
        size of the resulting array will be increased by one for each
        dimension.
        """

        # Create a deep copy of data
        data = copy.deepcopy(self.data)
        if add_zero_points:
            fpoints_max = self.fpoints_raw_max + 1
        else:
            fpoints_max = self.fpoints_raw_max
        for line in data:
            print(f'Getting raw points {line}')
        raw_data: list[list[MeasuredPoint|None]] = []
        # Fill array with existing data
        # Duplicate first line if necessary
        if add_zero_points:
            logger.info(f'{fpoints_max=}; {data[0].raw_points}')
            raw_data.append(
                copy.deepcopy(data[0].raw_data)
                + [copy.deepcopy(data[0].raw_data[-1])]*(fpoints_max - data[0].raw_points - 1) # type: ignore
                )
            first_point = copy.deepcopy(raw_data[0][0])
            first_point.pos = copy.deepcopy(data[0].startp) # type: ignore
            raw_data[0].insert(0, first_point)
        for i, line in enumerate(data):
            if add_zero_points:
                # raw_data is property and we need to copy in into local
                line_points = copy.deepcopy(line.raw_data)
                # For each line duplicate fisrt point and set its
                # position to line starting point
                first_point = copy.deepcopy(line_points[0])
                first_point.pos = copy.deepcopy(line.startp)
                line_points.insert(0, first_point)
                # Positions of all points are shifted up one slow step
                for point in line_points:
                    point.pos = point.pos + self.sstep
            else:
                line_points = copy.deepcopy(line.raw_data)
            raw_data.append(
                line_points
                + [copy.deepcopy(line_points[-1])]*(fpoints_max - len(line_points)) # type: ignore
                )
            a = line_points[-1]
            if i%2:
                raw_data[-1].reverse()

        # Fill remaining with None
        # for _ in range(self.spoints - len(self.data)):
        #     raw_data.append([None]*fpoints_max)
        logger.info(f'{len(raw_data)=}')
        for line in raw_data:
            print(f'{len(line)=}')
        # Format array to raw_data[x][y]
        if self.scan_dir[0] == 'H':
            # Reverse scan lines if start point on the right
            if self.scan_dir[1] == 'R':
                raw_data = [line[::-1] for line in raw_data]
            # Transpose for horizontal fast axis
            # raw_data = [list(x) for x in zip(*raw_data)]
            # Reverse line order if started from top
            if self.scan_dir[2] == 'T':
                raw_data.reverse()
        # Array format is already raw_data[x][y] for vertical fast scan
        else:
            # Reverse line order if scanned from right
            if self.scan_dir[1] == 'R':
                raw_data.reverse()
            # Reverse scan lines if scanned from top
            if self.scan_dir[2] == 'T':
                raw_data = [line[::-1] for line in raw_data]
        logger.info(f'Raw points shape {np.array(raw_data, dtype=object).shape}')
        return np.array(raw_data, dtype=object)
