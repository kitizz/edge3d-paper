import Nutmeg 0.1

Figure {
    Axis3D {
        id: ax3d
        handle: "ax"

        width: parent.width
        height: parent.height
        anchors.top: parent.top

        PointCloud {
            handle: "points"
            color: "#88FF4444"
            pointsize: 4
        }

        PointCloud {
            handle: "cluster"
            color: "#884444FF"
            pointsize: 8
        }
    }

}
