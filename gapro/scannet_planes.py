import json
import os
import sys
from ast import dump

import numpy as np
import torch


def isFourPointsInSamePlane(p0, p1, p2, p3, error):
    s1 = p1 - p0
    s2 = p2 - p0
    s3 = p3 - p0
    result = (
        s1[0] * s2[1] * s3[2]
        + s1[1] * s2[2] * s3[0]
        + s1[2] * s2[0] * s3[1]
        - s1[2] * s2[1] * s3[0]
        - s1[0] * s2[2] * s3[1]
        - s1[1] * s2[0] * s3[2]
    )
    if result - error <= 0 <= result + error:
        return True
    return False


def get_normal(quad_vert, center):
    tmp_A = []
    tmp_b = []
    for i in range(4):
        tmp_A.append([quad_vert[i][0], quad_vert[i][1], 1])  # x,y,1
        tmp_b.append(quad_vert[i][2])  # z
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    temp = A.T * A
    if np.linalg.det(temp) > 1e-10:
        fit = np.array(temp.I * A.T * b)
        a = fit[0][0] / fit[2][0]
        b = fit[1][0] / fit[2][0]
        c = -1.0 / fit[2][0]
        normal_vector = np.array([a, b, c])

        # print ("solution:%f x + %f y + %f z + 1 = 0" % (a, b, c) )

    else:  # vertical
        b = np.matrix([-1, -1, -1, -1]).T
        A = A[:, 0:2]
        temp = A.T * A
        fit = np.array(temp.I * A.T * b)
        a = fit[0][0]
        b = fit[1][0]
        c = 0
        normal_vector = np.array([a, b, c])
        # print ("solution:%f x + %f y + 1 = 0" % (a, b) )

    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector


def rectangle(quad_vert, center):
    """
    input: p1,p2,p3,p4
    return: normal vector, size, quad center, direction
    """

    quad_center = np.mean(quad_vert, axis=0)

    normal_vector = get_normal(quad_vert, center)

    vertical_normal_vector = np.array([normal_vector[0], normal_vector[1], 0])

    vertical_normal_vector = vertical_normal_vector / np.linalg.norm(vertical_normal_vector)

    edge_vector = quad_vert[0] - quad_vert[1]

    cos_theta = torch.cosine_similarity(torch.tensor(edge_vector), torch.tensor([0, 0, 1]), dim=0)

    l1 = np.linalg.norm(quad_vert[0] - quad_vert[1])
    l2 = np.linalg.norm(quad_vert[1] - quad_vert[2])
    l3 = np.linalg.norm(quad_vert[2] - quad_vert[3])
    l4 = np.linalg.norm(quad_vert[3] - quad_vert[0])
    l5 = (l1 + l3) / 2
    l6 = (l2 + l4) / 2

    if abs(cos_theta) > 0.5:
        h = np.array([l5])
        w = np.array([l6])
    else:
        h = np.array([l6])
        w = np.array([l5])

    rectangle = np.concatenate((quad_center, vertical_normal_vector, w, h))  # 3+3+2=8

    return rectangle


def get_center(verts):
    verts = np.array(verts)
    center = np.mean(verts, axis=0)
    return center


def get_box_from_quad(quad_vert, center):

    quad_center = np.mean(quad_vert, axis=0)

    normal_vector = get_normal(quad_vert, center)

    vertical_normal_vector = np.array([normal_vector[0], normal_vector[1], 0])

    vertical_normal_vector = vertical_normal_vector / np.linalg.norm(vertical_normal_vector)

    edge_vector = quad_vert[0] - quad_vert[1]

    cos_theta = torch.cosine_similarity(torch.tensor(edge_vector), torch.tensor([0, 0, 1]), dim=0)

    l1 = np.linalg.norm(quad_vert[0] - quad_vert[1])
    l2 = np.linalg.norm(quad_vert[1] - quad_vert[2])
    l3 = np.linalg.norm(quad_vert[2] - quad_vert[3])
    l4 = np.linalg.norm(quad_vert[3] - quad_vert[0])
    l5 = (l1 + l3) / 2
    l6 = (l2 + l4) / 2

    if abs(cos_theta) > 0.5:
        height = np.array([l5])
        width = np.array([l6])
    else:
        height = np.array([l6])
        width = np.array([l5])

    # rectangle = np.concatenate((quad_center,vertical_normal_vector,w,h))  #3+3+2=8

    vertical_normal_vector = vertical_normal_vector / max(np.linalg.norm(vertical_normal_vector), 1e-6)
    quad_center = np.array(quad_center)
    vertical_normal_vector = np.array(vertical_normal_vector)
    x1 = quad_center[0] + width * vertical_normal_vector[1] / 2
    x2 = quad_center[0] - width * vertical_normal_vector[1] / 2

    # x=[x1,x2]

    y1 = quad_center[1] - width * vertical_normal_vector[0] / 2
    y2 = quad_center[1] + width * vertical_normal_vector[0] / 2

    # y=[y1,y2]

    h1 = quad_center[2] + height / 2
    h2 = quad_center[2] - height / 2

    corners = []

    for h in [h1, h2]:
        corners.append([x1, y1, h])
        corners.append([x2, y2, h])

    corners = np.array([corners])  # 2 x 3

    bounding_box = np.array([min(x1, x2), min(y1, y2), min(h1, h2), max(x1, x2), max(y1, y2), max(h1, h2)])[:, 0]

    # bounding_box[:3] -= 0.1
    # bounding_box[3:] += 0.1
    return bounding_box


def transform(scan_name, mesh_vertices):
    meta_file = "dataset/scannetv2/scans_transform/" + os.path.join(scan_name, scan_name + ".txt")
    lines = open(meta_file).readlines()
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]
    return mesh_vertices


def get_wall_boxes(scan_name):
    quad_file_path = "dataset/scannetv2/scannet_planes/" + scan_name + ".json"
    # return [], [], []
    if not os.path.exists(quad_file_path):
        return [], [], []
    # print(quad_file_path)
    with open(quad_file_path, "r") as quad_file:
        plane_dict = json.load(quad_file)
    quad_dict = plane_dict["quads"]
    # total_quad_num = len(quad_dict)

    vert_dict = plane_dict["verts"]

    for i in range(0, len(vert_dict)):
        temp = vert_dict[i][1]
        vert_dict[i][1] = -vert_dict[i][2]
        vert_dict[i][2] = temp

    verts = np.array(vert_dict)

    verts = transform(scan_name, verts)

    quads = [i for i in quad_dict if len(i) == 4]

    quad_verts = np.asarray([[verts[j] for j in _] for _ in quads])

    quad_verts_filter_ = np.asarray(
        [
            quad_vert
            for quad_vert in quad_verts
            if isFourPointsInSamePlane(quad_vert[0], quad_vert[1], quad_vert[2], quad_vert[3], 100)
        ]
    )

    room_center = get_center(vert_dict)  # room center

    quad_verts_filter = np.asarray(
        [quad_vert for quad_vert in quad_verts_filter_ if abs(get_normal(quad_vert, room_center)[2]) < 0.2]
    )  # only vertical

    # horizontal_quads = np.asarray([quad_vert for quad_vert in quad_verts_filter_
    #                                     if abs(get_normal(quad_vert, room_center)[2])>0.8]) #only horizontal

    # rectangles = np.array([rectangle(_, room_center) for _ in quad_verts_filter])
    if len(quad_verts_filter) == 0:
        return [], [], []

    boxes = np.array([get_box_from_quad(quad, room_center) for quad in quad_verts_filter])  # N x 6
    cls = np.ones(boxes.shape[0]).astype(np.int) * 18
    volumes = np.prod(np.clip((boxes[:, 3:] - boxes[:, :3]), a_min=0.0, a_max=None), axis=-1)  # N

    # breakpoint()
    # print(boxes[:, 3:] - boxes[:, :3])
    return cls, boxes, volumes
