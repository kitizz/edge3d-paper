import Nutmeg.Gui 0.1
import QtQuick 2.5

Gui {
    Column {
        width: parent.width

        Row {
            Button {
                handle: "pprev"
                text: "<<"
            }
            Button {
                handle: "prev"
                text: "<"
            }
            Button {
                handle: "next"
                text: ">"
            }
            Button {
                handle: "nnext"
                text: ">>"
            }
        }

        Button {
            handle: "nextframe"
            text: "Next Frame"
        }
    }
}
