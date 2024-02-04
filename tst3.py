    def get_raw_points(self, add_zero_points: bool=False) -> np.ndarray:
        """
        2D array containing deep copy of measured points.

        The first index is vertical axis, the second is horizontal.
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

        tstart = time.time()

        icashed = len(self._raw_points)
        # Create a deep copy of new data
        data = copy.deepcopy(self.data[icashed:])
        logger.info(f'copy all data in {(time.time() - tstart):.3}')
        tcur = time.time()
        # Array for result
        raw_data: list[list[MeasuredPoint]] = []
        # Fill array with existing data
        for i, line in enumerate(data):
            line_points = copy.deepcopy(line.raw_data)
            # To match line size last point in each line is duplicated
            # required times
            last_point = copy.deepcopy(line_points[-1])
            while (j:=self.fpoints_raw_max - len(line_points)) > 0:
                line_points.append(copy.deepcopy(last_point))
                j -= 1
            # For plotting data duplicate first scan line and shift
            # other lines by one slow scan step
            if add_zero_points:
                # For each line the first position is starting point of
                # that line
                first_point = copy.deepcopy(line_points[0])
                first_point.pos = line.startp
                line_points.insert(0, first_point)
                if i == 0:
                    raw_data.append(copy.deepcopy(line_points))
                for point in line_points:
                    point.pos = point.pos + self.sstep
            raw_data.append(copy.deepcopy(line_points))
            logger.info(f'before line sort {(time.time() - tcur):.3}')
            tcur = time.time()
            # Sort points within each scan line
            raw_data[-1].sort(key=lambda x: getattr(x.pos, self.faxis))
            logger.info(f'line sort {(time.time() - tcur):.3}')
            tcur = time.time()
        # Sort lines
        raw_data.sort(key=lambda x: getattr(x[0].pos, self.saxis))
        logger.info(f'all lines sort {(time.time() - tcur):.3}')
        tcur = time.time()
        # Transpose data if fast scan axis is vertical
        if self.scan_dir[0] == 'V':
            raw_data = [list(x) for x in zip(*raw_data)]

        logger.info(f'get_raw_points in {(time.time() - tstart):.3}')
        return np.array(raw_data, dtype=object)
