import Nutmeg 0.1

Figure {
    Axis3D {
        id: ax3d
        handle: "ax3d"

        width: parent.width
        height: parent.height
        anchors.top: parent.top

        RayCloud {
            handle: "rays"
            color: "#884444FF"
            linewidth: 1
        }
    }

    // Axis {
    //     handle: "ax"

    //     // anchors.fill: parent
    //     anchors.top: ax3d.bottom
    //     anchors.bottom: parent.bottom
    //     width: parent.width
    //     // height: parent.height/2

    //     LineSegmentPlot {
    //         handle: "rays"
    //         line { width: 2; color: "#884444FF" }
    //     }
    // }
}
