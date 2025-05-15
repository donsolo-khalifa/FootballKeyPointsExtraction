using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using System.Threading;


public class AnimationCode : MonoBehaviour {
    public Transform[] Body;   // size=33
    public Transform Ball;
    List<string> lines;
    int counter;
    void Start() {
        lines = System.IO.File.ReadLines("Assets/AnimationFile.txt").ToList();

    }
    void Update() {
        if (lines.Count == 0) return;
        var vals = lines[counter].Split(',');
        for (int i = 0; i < 33; i++) {
            float x = float.Parse(vals[i * 3 + 0]);
            float y = float.Parse(vals[i * 3 + 1]);
            float z = float.Parse(vals[i * 3 + 2]);
            Body[i].localPosition = new Vector3(x, y, z);
        }
        float bx = float.Parse(vals[99]), by = float.Parse(vals[100]), bz = float.Parse(vals[101]);
        Ball.localPosition = new Vector3(bx, by, bz);
        counter = (counter + 1) % lines.Count;
        Thread.Sleep(30);

    }
}
