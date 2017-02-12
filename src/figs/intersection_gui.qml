import Nutmeg.Gui 0.1
import QtQuick 2.5

Gui {
    Column {
        width: parent.width

        Slider {
            id: frame
            handle: "frame"
            text: "Frame"
            minimumValue: 0
            maximumValue: 4000
            stepSize: 1
            decimals: 0
        }

        Item {
            width: parent.width
            height: prev.height*1.5

            Button {
                id: prev
                handle: "prevframe"
                text: "<"
                onClicked: frame.value -= frame.stepSize
                anchors.left: parent.left
            }
            Button {
                handle: "nextframe"
                text: ">"
                onClicked: frame.value += frame.stepSize
                anchors.right: parent.right
            }
        }

        Slider {
            id: segment
            handle: "segment"
            text: "Segment"
            minimumValue: 0
            maximumValue: 0
            stepSize: 1
            decimals: 0
        }
        Item {
            width: parent.width
            height: prev.height*1.5

            Button {
                handle: "prevseg"
                text: "<"
                onClicked: segment.value -= 1
                anchors.left: parent.left
            }
            Button {
                handle: "nextseg"
                text: ">"
                onClicked: segment.value += 1
                anchors.right: parent.right
            }
        }

        Slider {
            handle: "frameoffset"
            text: "Frame Offset"
            minimumValue: -20
            maximumValue: 20
            stepSize: 1
            decimals: 0
        }

        Slider {
            handle: "index"
            text: "Index"
            minimumValue: 0
            maximumValue: 0
            stepSize: 1
            decimals: 0
        }

        Slider {
            handle: "anglesupport"
            text: "Angle Support"
            minimumValue: 0
            maximumValue: 1000
            stepSize: 1
            decimals: 0
        }

        Slider {
            handle: "planarsupport"
            text: "Planar Support"
            minimumValue: 0
            maximumValue: 1000
            stepSize: 1
            decimals: 0
        }

        Slider {
            handle: "framesupport"
            text: "Frame Support"
            minimumValue: 0
            maximumValue: 0
            stepSize: 1
            decimals: 0
        }

        Button {
            handle: "cachebtn"
            text: "Cache Segment"
        }

        Button {
            handle: "exportbtn"
            text: "Export .ply segments"
        }
    }
}
