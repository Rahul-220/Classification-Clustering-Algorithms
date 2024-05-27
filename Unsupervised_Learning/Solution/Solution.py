#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import os
import random
import matplotlib.pyplot as plt

def read_file(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        data = []
        # Read each line in the file and remove the last column
        for line in file:
            line_data = list(map(float, line.strip().split()))
            data.append(line_data[:-1])
    return data

# Calculating the initial centroids:
def initial_centroids(number_of_clusters, data_points):
    random.seed(0)
    centroids = random.sample(data_points, number_of_clusters)
    clusters, data_points = get_clusters(number_of_clusters, data_points, centroids)
    return clusters, data_points
    
# Updating clusters for each iteration:
def get_clusters(number_of_clusters, data_points, centroids):
    clusters = [[] for _ in range(number_of_clusters)]
    for data_point in data_points:
        distances = []
        for centroid in centroids:
            distances.append(calculate_euclidean_distance(data_point, centroid))
        index = np.argmin(distances)
        clusters[index].append(data_point)
    return clusters, data_points


# Updating Centroids:
def update_centroids(k, no_of_dimensions, clusters):
    centroids = [[0] * no_of_dimensions for _ in range(k)]
            
    for i in range(k):
        if len(clusters[i]) > 0:
            for index in range(len(clusters[i])):
                 for dimension in range(no_of_dimensions):
                    centroids[i][dimension] += clusters[i][index][dimension]
        
    for i in range(k):
        if len(clusters[i]) > 0:
            for dimension in range(no_of_dimensions):
                centroids[i][dimension] /= len(clusters[i])
    return centroids

# Calculating SSE:
def get_sse(k, clusters, centroids):
    error = []
    for i in range(k):
        if len(clusters[i]) > 0:
            sse_distance = []
            for index in range(len(clusters[i])):
                sse_distance.append(calculate_euclidean_distance(clusters[i][index], centroids[i]))
            error.append(sum(sse_distance))
    sum_of_error = sum(error)
    return sum_of_error

# Calculating Euclidean Distance:
def calculate_euclidean_distance(pt1, pt2):
    differences = []
    squares = []
    for i in range(len(pt1)):
        differences.append(pt1[i] - pt2[i])
    for difference in differences:
        squares.append(difference ** 2)
    sum_squares = sum(squares)
    total_distance = sum_squares ** 0.5
    return total_distance


# Plotting the results:
def plotGraph(no_of_clusters, sse_values):
    plt.plot(list(no_of_clusters), sse_values, marker='o', color='red', linestyle='-')
    plt.xlabel("K Values")
    plt.ylabel("SSE Values")
    plt.title(f"K-Means Clustering: SSE Values vs No: of Clusters")
    plt.grid(True)
    plt.show()

        
    
def main():
    file_path = sys.argv[1]
    if os.path.exists(file_path):
        data = read_file(file_path)
    else:
        print("File doesn't exist.")
        
    no_of_clusters = range(2, 11)
    sse_values =[]
    no_of_iterations = 20
    
    for k in no_of_clusters:
        clusters, data_points = initial_centroids(k, data)
        no_of_dimensions = len(data_points[0])
        
        for _ in range(no_of_iterations):
            centroids = update_centroids(k, no_of_dimensions, clusters)
            clusters, data_points = get_clusters(k, data_points, centroids)
        
        sse = get_sse(k, clusters, centroids)
        print(f"For k = {k} After {no_of_iterations} iterations: Error = {sse:.4f}")
        sse_values.append(sse)
        
    plotGraph(no_of_clusters, sse_values)
        
if __name__ == "__main__":
    main()
        
        



# In[ ]:




