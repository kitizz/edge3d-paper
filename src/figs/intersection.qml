import Nutmeg 0.1

Figure {
    id: fig
    property real midx: 0.5*width
    property real midy: 0.5*height
    // Grid {
    //     columns: 2

    //     anchors.fill: parent

    Axis {
        handle: "ax"
        aspectRatio: 1
        width: fig.midx
        height: fig.midy

        ImagePlot {
            handle: "im"
        }

        LinePlot {
            handle: "rays"
            line { width: 2; color: "#33BB33" }
        }

        LinePlot {
            handle: "P0"
            line { width: 4; color: "#FF5555"; style: "." }
        }
        LinePlot {
            handle: "P1"
            line { width: 4; color: "#55AAFF"; style: "." }
        }
    }

    Axis {
        handle: "fit"
        y: fig.midy
        width: fig.midx
        height: fig.height - fig.midy

        aspectRatio: 1
        // shareX: "2dx"
        // shareY: "2dy"

        ImagePlot {
            handle: "im"
        }

        LineSegmentPlot {
            handle: "rays"
            line { width: 1; color: "#114444FF"; style: "-" }
        }

        LineSegmentPlot {
            handle: "rays2"
            line { width: 1; color: "#33FF4444"; style: "-" }
        }

        LineSegmentPlot {
            handle: "rays3"
            line { width: 1; color: "#6644FF44"; style: "-" }
        }

        LinePlot {
            handle: "l0"
            line { width: 4; color: "#FF5555"; style: "-" }
        }

        LinePlot {
            handle: "P2"
            line { width: 4; color: "#33CC33"; style: "." }
        }

        LinePlot {
            handle: "P0"
            line { width: 2; color: "#445555FF"; style: "." }
        }

        LinePlot {
            handle: "P1"
            line { width: 2; color: "#FF5555"; style: "." }
        }


        LinePlot {
            handle: "l1"
            line { width: 4; color: "#AA33FF33"; style: "-" }
        }
    }

    Axis {
        handle: "tangent"
        x: fig.midx
        width: fig.width - fig.midx
        height: fig.midy

        aspectRatio: 1
        shareX: "2dx"
        shareY: "2dy"

        LineSegmentPlot {
            handle: "rays"
            line { width: 1; color: "#114444FF"; style: "-" }
        }

        LinePlot {
            handle: "l0"
            line { width: 2; color: "#FF5555"; style: "-" }
        }

        LinePlot {
            handle: "P0"
            line { width: 3; color: "#FF5555"; style: "." }
        }

        LinePlot {
            handle: "l1"
            line { width: 4; color: "#AA33FF33"; style: "-" }
        }
    }

    Axis {
        handle: "normal"
        x: fig.midx
        y: fig.midy
        width: fig.width - fig.midx
        height: fig.height - fig.midy

        aspectRatio: 1
        shareX: "2dx"
        shareY: "2dy"

        LineSegmentPlot {
            handle: "rays"
            line { width: 1; color: "#114444FF"; style: "-" }
        }

        LinePlot {
            handle: "l0"
            line { width: 2; color: "#FF5555"; style: "-" }
        }

        LinePlot {
            handle: "P0"
            line { width: 3; color: "#FF5555"; style: "." }
        }

        LinePlot {
            handle: "circle"
            line { width: 4; color: "#AA33FF33"; style: "--" }
        }
    }
    // }
}
