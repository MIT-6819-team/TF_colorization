import ujson, gzip

class DataLoader(object):
    def __init__(self):
        f = gzip.open('saturation_values.json.gz', 'rt')
        saturation_index = ujson.load(f)
        saturation_index['n03982430/n03982430_22677.JPEG']

        self.current_datapoint_index = 0

    def next_batch(self, batch_size):
        x_batch = np.zeros((batch_size, INPUT_SIZE))
        y__batch = np.zeros((batch_size, 1))

        for i in range(batch_size):
            if random_select:
                x, y_ = self.all_datapoints[int(random.random() * len(self.all_datapoints))]
            else:
                x, y_ = self.all_datapoints[self.current_datapoint_index]

            x_batch[i, ...] = x
            y__batch[i, ...] = y_

            if not random_select:
                self.current_datapoint_index += 1
                if self.current_datapoint_index >= len(self.all_datapoints):
                    self.current_datapoint_index = 0

        return x_batch, y__batch
