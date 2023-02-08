using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MainCharacter : MonoBehaviour
{
    private CharacterController controller;
    private Vector3 direction;
    private Vector3 position;
    public float forwardSpeed;

    private int desierdLane = 1; //By default is the middle lane; 
    public float laneDistance = 4; //Distance between two lanes
    // Start is called before the first frame update
    void Start()
    {
        controller = GetComponent<CharacterController>();
    }

    // Update is called once per frame
    void Update()
    {
        direction.z = forwardSpeed;
        position.y = 0;

        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            desierdLane++;
            if (desierdLane == 3)
            {
                desierdLane = 2;
            }
        }

        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            desierdLane--;
            if (desierdLane == -1)
            {
                desierdLane = 0;
            }
        }

            Vector3 targetPosition = transform.position.z * transform.forward + transform.position.y * transform.up;

        if (desierdLane == 0)
        {
            targetPosition += Vector3.left * laneDistance;
        }
        if (desierdLane == 2)
        {
            targetPosition += Vector3.right * laneDistance;
        }

        transform.position = Vector3.Lerp(transform.position,targetPosition,1000*Time.deltaTime);
    }

    private void FixedUpdate()
    {
        controller.Move(direction * Time.fixedDeltaTime);
    }
}
