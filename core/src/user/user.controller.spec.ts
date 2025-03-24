import { Test, TestingModule } from '@nestjs/testing';
import { UserController } from './user.controller';
import { UserService } from './user.service';

describe('UserController', () => {
  let controller: UserController;
  let service: UserService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UserController],
      providers: [
        {
          provide: UserService,
          useValue: {
            findAll: jest.fn().mockResolvedValue([{ id: 1, name: 'John Doe' }]),
            findOne: jest.fn().mockResolvedValue({ id: 1, name: 'John Doe' }),
            create: jest.fn().mockResolvedValue({ id: 2, name: 'Jane Doe' }),
            update: jest.fn().mockResolvedValue({ id: 1, name: 'Updated User' }),
            remove: jest.fn().mockResolvedValue({ success: true }),
          },
        },
      ],
    }).compile();

    controller = module.get<UserController>(UserController);
    service = module.get<UserService>(UserService);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  it('should return a list of users', async () => {
    const users = await controller.findAll();
    expect(users).toEqual([{ id: 1, name: 'John Doe' }]);
  });

  it('should return a single user by ID', async () => {
    const user = await controller.findOne(1);
    expect(user).toEqual({ id: 1, name: 'John Doe' });
  });

  it('should create a new user', async () => {
    const newUser = await controller.create({ name: 'Jane Doe' });
    expect(newUser).toEqual({ id: 2, name: 'Jane Doe' });
  });

  it('should update an existing user', async () => {
    const updatedUser = await controller.update(1, { name: 'Updated User' });
    expect(updatedUser).toEqual({ id: 1, name: 'Updated User' });
  });

  it('should delete a user', async () => {
    const result = await controller.remove(1);
    expect(result).toEqual({ success: true });
  });
});
