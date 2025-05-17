using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Material))]
public class SkeletonRenderer : MonoBehaviour {
    [Tooltip("Parent GameObject whose immediate children are the 33 joint Transforms (in order).")]
    public GameObject Body;

    [Tooltip("Prefab or default LineRenderer settings (assign a GameObject with a LineRenderer).")]
    public LineRenderer linePrefab;

    // MediaPipe�s 33?landmark connections
    // Each pair is (startIndex, endIndex)
    private readonly int[,] _connections = new int[,]
    {
        {0,1}, {1,2}, {2,3}, {3,7},        // right arm
        {0,4}, {4,5}, {5,6}, {6,8},        // left arm
        {9,10},                             // shoulders
        {11,12}, {12,14}, {14,16},         // right leg
        {11,13}, {13,15}, {15,17},         // left leg
        {11,23}, {12,24},                  // hips to shoulders
        {23,24}, {23,25}, {24,26},         // hips
        {25,27}, {27,29},                  // right lower leg
        {26,28}, {28,30},                  // left lower leg
        {29,31}, {30,32}, {32,28}, {31,27}                    // feet
        // add more pairs if desired (e.g. face or hands)
    };

    private Transform[] _joints;            // bodyRoot children
    private List<LineRenderer> _lines;      // one per connection

    void Awake() {
        // grab all joint transforms
        int n = Body.transform.childCount;
        _joints = new Transform[n];
        for (int i = 0; i < n; i++)
            _joints[i] = Body.transform.GetChild(i);

        // instantiate a LineRenderer for each connection
        int connCount = _connections.GetLength(0);
        _lines = new List<LineRenderer>(connCount);
        for (int i = 0; i < connCount; i++) {
            var lr = Instantiate(linePrefab, transform);
            lr.positionCount = 2;
            // optional: tweak widths here
            lr.startWidth = 0.05f;
            lr.endWidth = 0.05f;
            _lines.Add(lr);
        }
    }

    void Update() {
        // update each line
        for (int i = 0; i < _lines.Count; i++) {
            int startIdx = _connections[i, 0];
            int endIdx = _connections[i, 1];

            Vector3 p0 = _joints[startIdx].position;
            Vector3 p1 = _joints[endIdx].position;

            _lines[i].SetPosition(0, p0);
            _lines[i].SetPosition(1, p1);
        }
    }
}
