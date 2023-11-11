import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import csv

# input: a String with the path to the .csv file
# output: a list of dicts, each line representing one dict
def load_data(filepath):

    f = open(filepath)

    #each .csv line has 7 components:
    #Country,Population,Net migration,GDP ($ per capita),Literacy (%),Phones (per 1000),Infant mortality (per 1000 births)

    reader = csv.reader(f)
    dict_list = []
    header_reached = False

    for row in reader:

        #skip the header line
        if (not header_reached):
            header_reached = True
            continue

        d = {
            '' : row[0] ,
            'Country' : row[1],
            'Population' : row[2],
            'Net migration' : row[3],
            'GDP ($ per capita)' : row[4],
            'Literacy (%)' : row[5],
            'Phones (per 1000)' : row[6],
            'Infant mortality (per 1000 births)' : row[7]
            }

        dict_list.append(d)

    return dict_list

# input: a single dict
# output: a (6,) size numpy array representing a country
def calc_features(row):
    x1 = row['Population']
    x2 = row['Net migration']
    x3 = row['GDP ($ per capita)']
    x4 = row['Literacy (%)']
    x5 = row['Phones (per 1000)']
    x6 = row['Infant mortality (per 1000 births)']

    arr = np.array([x1, x2, x3, x4, x5, x6])

    return arr.astype(np.float64)

# input: a list size n of size (6,) numpy arrays
# output: a (n-1, 4) size numpy array, where for each row i, 
#         Z[i, 0] and Z[i, 1] represent the indices of the two clusters merged, 
#         Z[i, 2] is the complete linkage distance between the merged clusters
#         Z[i, 3] is the size of the new cluster formed (i.e. num of countries)
def hac(features):

    # how many initial clusters exist (fixed)
    n = len(features)

    # list representing the current cluster indices
    clusters = []
    for i in range(len(features)):
        clusters.append(i)

    # make the distance matrix for the original clusters
    distance = np.zeros([len(features), len(features)])
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            d = np.linalg.norm(features[i] - features[j])
            distance[i, j] = d
            distance[j, i] = d

    output = np.zeros([n-1, 4])

    for a in range(len(output)):

        i = 0
        active_clusters = len(clusters)

        min_distance = 2**31-1
        min_first_idx = -1
        min_second_idx = -1
        first_cluster_size = -1
        second_cluster_size = -1

        # check every cluster against each other:
        while (i < active_clusters):

            j = i + 1
            while (j < active_clusters):
                first_cluster = []

                # grabs the ith value from clusters
                first_cluster.append(clusters[i])
                orig_first_idx = clusters[i]

                # and loops down until first_cluster contains
                # only the indices of the original vectors in the ith cluster
                switch = contains_nonstandard_cluster(first_cluster, n)

                while (switch):
                    for element in first_cluster:
                        if (element >= n):
                            Z_index = int(element - n)
                            idx_1 = output[Z_index, 0]
                            idx_2 = output[Z_index, 1]
                            first_cluster.remove(element)
                            first_cluster.append(idx_1)
                            first_cluster.append(idx_2)
                    switch = contains_nonstandard_cluster(first_cluster, n)

                # do the same for the second cluster
                second_cluster = []
                second_cluster.append(clusters[j])
                orig_second_idx = clusters[j]

                switch = contains_nonstandard_cluster(second_cluster, n)
                while (switch):
                    for element in second_cluster:
                        if (element >= n):
                            Z_index = int(element - n)
                            idx_1 = output[Z_index, 0]
                            idx_2 = output[Z_index, 1]
                            second_cluster.remove(element)
                            second_cluster.append(idx_1)
                            second_cluster.append(idx_2)
                    switch = contains_nonstandard_cluster(second_cluster, n)

                # now first_cluster and second_cluster should only contain
                # the indices of the vectors that their cluster contains
                dist = complete_linkage_distance(features, first_cluster, second_cluster)

                if (dist < min_distance):
                    min_distance = dist
                    first_cluster_size = len(first_cluster)
                    second_cluster_size = len(second_cluster)
                    if (orig_first_idx < orig_second_idx):
                        min_first_idx = orig_first_idx
                        min_second_idx = orig_second_idx
                    else:
                        min_first_idx = orig_second_idx
                        min_second_idx = orig_first_idx

                # if tie on distance, sort by first index
                elif (dist == min_distance):
                    first_cluster_size = len(first_cluster)
                    second_cluster_size = len(second_cluster)
                    if (orig_first_idx < orig_second_idx):
                        if (orig_first_idx < min_first_idx):
                            min_first_idx = orig_first_idx
                            min_second_idx = orig_second_idx

                        # if tie on first index, sort by second index
                        elif (orig_first_idx == min_first_idx):
                            if (orig_first_idx < orig_second_idx):
                                if (orig_second_idx < min_second_idx):
                                    min_second_idx = orig_second_idx
                            else:
                                if (orig_first_idx < min_second_idx):
                                    min_second_idx = orig_first_idx

                    else:
                        if (orig_second_idx < min_first_idx):
                            min_first_idx = orig_second_idx
                            min_second_idx = orig_first_idx

                        # if tie on first index, sort by second index
                        elif (orig_second_idx == min_first_idx):
                            if (orig_first_idx < orig_second_idx):
                                if (orig_second_idx < min_second_idx):
                                    min_second_idx = orig_second_idx
                            else:
                                if (orig_first_idx < min_second_idx):
                                    min_second_idx = orig_first_idx

                j += 1

            i += 1

        # update the list of active clusters

        clusters.remove(min_first_idx)
        clusters.remove(min_second_idx)
        clusters.append(n+a)

           # and the size (to keep while loops in check)
        active_clusters = len(clusters)

        output[a, 0] = int(min_first_idx)
        output[a, 1] = int(min_second_idx)

        # and fill out the rest of the current row
        output[a, 2] = min_distance
        output[a, 3] = int(first_cluster_size + second_cluster_size)

    return output
                

# input: two lists, each representing a cluster 
# output: a tuple containing two ints representing
# the two indices of the vectors furthest apart, 
# using complete linkage, then a third int representing
# the distance between those two vectors
def complete_linkage_distance(features, list_1, list_2):

    max_distance = -1       # max distance recorded between two vectors

    for i in range(len(list_1)):
        for j in range(len(list_2)):
            distance = np.linalg.norm(features[int(list_1[i])] - features[int(list_2[j])])
            if (distance > max_distance):
                    max_distance = distance

    return max_distance

# checks if a cluster that we check contains any indices corresponding to one of the unoriginal clusters
def contains_nonstandard_cluster(lst, threshold):
    for element in lst:
        if element >= threshold:
            return True
    return False

# visualizes the hierarchical agglomerative clustering on the country's feature representation
def fig_hac(Z, names):

    fig = plt.figure()

    dendro = sp.cluster.hierarchy.dendrogram(Z, labels=names, leaf_rotation=50)

    fig.tight_layout()

    return fig

# Takes a list of feature vectors and computes the normalized values. The output
# should be a list of normalized feature vectors in the same format as the input.
def normalize_features(features):

    x1_values = []
    x2_values = []
    x3_values = []
    x4_values = []
    x5_values = []
    x6_values = []
    for row in features:
        x1_values.append(row[0])
        x2_values.append(row[1])
        x3_values.append(row[2])
        x4_values.append(row[3])
        x5_values.append(row[4])
        x6_values.append(row[5])        

    x1_mean = np.mean(x1_values)
    x2_mean = np.mean(x2_values)
    x3_mean = np.mean(x3_values)
    x4_mean = np.mean(x4_values)
    x5_mean = np.mean(x5_values)
    x6_mean = np.mean(x6_values)
    x1_std = np.std(x1_values)
    x2_std = np.std(x2_values)
    x3_std = np.std(x3_values)
    x4_std = np.std(x4_values)
    x5_std = np.std(x5_values)
    x6_std = np.std(x6_values)

    new_features = []

    for row in features:
        new_row = []
        new_row.append((row[0] - x1_mean)/x1_std)
        new_row.append((row[1] - x2_mean)/x2_std)
        new_row.append((row[2] - x3_mean)/x3_std)
        new_row.append((row[3] - x4_mean)/x4_std)
        new_row.append((row[4] - x5_mean)/x5_std)
        new_row.append((row[5] - x6_mean)/x6_std)
        new_features.append(np.asarray(new_row))

    return new_features

def main():
    data = load_data('countries.csv')
    country_names = [row['Country'] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = 50
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    fig = fig_hac(Z_normalized, country_names[:n])
    plt.show()

    print(Z_normalized)

main()
