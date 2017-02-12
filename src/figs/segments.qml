import Nutmeg 0.1

Figure {
    Axis {
        handle: "ax"
        aspectRatio: 1

        ImagePlot {
            handle: "im"
        }

        LinePlot {
            handle: "P0"
            line { width: 4; color: "#FF5555"; style: "." }
        }
        LinePlot {
            handle: "P1"
            line { width: 4; color: "#55AAFF"; style: "." }
        }

        LineSegmentPlot {
            handle: "seg"
            line { width: 1; color: "#444444FF"; style: "-" }
        }
    }
}
